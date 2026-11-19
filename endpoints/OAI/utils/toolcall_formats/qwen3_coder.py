import re
import json
from common.logger import xlogger
from endpoints.OAI.types.tools import ToolCall, Tool
from endpoints.OAI.utils.toolcall_formats.common import FormatSignature, coerce_param_value

"""
Qwen3.5 / Qwen3-Coder - pseudo-XML syntax

Raw format:
    <tool_call>
        <function=__FUNCTION_NAME__>
            <parameter=__PARAMETER_NAME_1__>
                __PARAMETER_1__
            </parameter>
            <parameter=__PARAMETER_NAME_2__>
                __PARAMETER_2__
            </parameter>
            ...
        </function>
    </tool_call>
"""

# TODO: the outer <tool_call> wrapper is supposedly optional in some deployments; the parser
#   handles both, but detecting tool calls in the stream currently relies on <tool_call> being
#   emitted by the model.

TOOLCALL_START = "<tool_call>"
TOOLCALL_END = "</tool_call>"

DETECT = FormatSignature(
    template_markers=("<tool_call>", "<function="),
    special_tokens=("<tool_call>", "</tool_call>"),
    architectures=("Qwen3_5", "Qwen3Next", "Qwen4", "Qwen3Coder", "Step3", "NemotronH"),
    reasoning_tags=("<think>", "</think>"),
)

# Tag constants: a literal close tag is only a boundary when boundary_state()
# allows what follows it, so these are compared against the text, not used as regexes.
CALL_CLOSE = "</" + "tool_call>"
FUNC_CLOSE = "</" + "function>"
PARAM_CLOSE = "</" + "parameter>"
FUNC_OPEN = "<" + "function="
PARAM_OPEN = "<" + "parameter="

# Open-tag patterns for the boundary scan. The outer level has no name, so its
# pattern carries no capture group; _scan_once reads a name only when there is one.
_CALL_OPEN_SCAN = re.compile("<" + "tool_call>")
_FUNC_OPEN_SCAN = re.compile("<" + "function=([^>" + chr(92) + "s]+)[^>]*>")
_PARAM_OPEN_SCAN = re.compile("<" + "parameter=([^>" + chr(92) + "s]+)[^>]*>")


# ─── Boundary scan (tabbyAPI-463) ─────────────────────────────────────────────
# WHY: this parser matched each block with a non-greedy regex and took the FIRST
# close tag. An argument value that has to carry literal call markup - the shape
# every coding agent produces when it writes a file, a shell command or a prompt
# that quotes this wire format - therefore ended the value early, and the tail of
# that value plus every LATER parameter vanished with no error anywhere:
# finish_reason=tool_calls, a success result, a silently truncated payload.
# Census, repro and the failed template-side alternative:
# Launchers/research/litellm-tool-call-integrity (F1, F8, F10). Measured on the
# live Flash Next EXL3 engine: 16/16 delivered calls corrupted, and no instruction
# in the chat template prevents it, because the cut happens here, after generation
# has already ended.
#
# RULE: a close tag ends a parameter value only when what follows it, after
# whitespace, is something this format allows in that position; anything else is
# literal value content and the scan moves past it. Same-kind open tags balance,
# so a value may carry a whole echoed call of this format. The call and function
# levels have no continuation table to consult (their body is markup, not free
# text) and are read by balance alone. No request schema is needed - the dialect's
# own grammar decides the boundary - which is what keeps every existing call site
# (tools.parse_toolcalls, the #462 delta streamer's cross-checks) working
# unchanged, and what the delta streamer calls into as well: one rule, both paths.

# What may follow a parameter close tag and still be a boundary: another parameter
# of the same call, the end of the call (the function close, or the wrapper close
# that TagStreamParser keeps in the tool channel), or the parameter close of an
# OUTER block - the shape where a value quotes a whole call. Anything else is
# prose, so the tag is literal value content.
_AFTER_PARAM_CLOSE = (PARAM_CLOSE, FUNC_CLOSE, PARAM_OPEN, CALL_CLOSE)


def boundary_state(rest, legal, at_end=True):
    """Classify what follows a close tag: "boundary", "literal" or "unknown".

    "boundary" closes the block, "literal" means the close tag is value content,
    "unknown" means more text is needed to tell - only a streaming reader can get
    that, because for a one-pass parser the text it holds IS the whole text.
    The delta streamer calls this with at_end=False so it can hold a decision
    instead of guessing (toolcall_stream.py); both readers therefore apply one
    rule, which is what keeps streamed fragments byte-identical to the parse.

    The prefix table must never contain the empty string: startswith("") matches
    everything and would make every close tag a boundary, which is the exact bug
    this scan replaces.
    """

    if legal is None:
        return "boundary"               # this level closes on its own tag
    stripped = rest.lstrip("\r\n \t")
    if stripped == "":
        return "boundary" if at_end else "unknown"
    if any(stripped.startswith(prefix) for prefix in legal):
        return "boundary"
    if not at_end and any(prefix.startswith(stripped) for prefix in legal):
        # Too little text to decide: what arrived is still a prefix of something
        # legal ("", "<", "</p"...). A streaming reader must hold the decision
        # until the next chunk; a one-pass parser has all the text there is.
        return "unknown"
    return "literal"


def _scan_once(text, start, open_re, close_tag, legal, nest):
    """Read one block. Returns (name, body, next_pos, open_start, unclosed).

    See _scan for the rule; `nest` toggles whether a same-kind open tag inside
    the block raises a nesting depth.
    """
    m = open_re.search(text, start)
    if not m:
        return None
    name = m.group(1) if open_re.groups else None
    body_start, open_start = m.end(), m.start()
    depth = 0
    cursor = body_start
    while True:
        close = text.find(close_tag, cursor)
        nxt = open_re.search(text, cursor)
        if nest and nxt is not None and (close < 0 or nxt.start() < close):
            depth += 1                      # a nested block of the same kind
            cursor = nxt.end()
            continue
        if close < 0:                       # nothing ever closed this block
            return name, "", len(text), open_start, True
        if depth > 0:                       # closes a nested block, not this one
            depth -= 1
            cursor = close + len(close_tag)
            continue
        if boundary_state(text[close + len(close_tag):], legal) == "boundary":
            return name, text[body_start:close], close + len(close_tag), open_start, False
        cursor = close + len(close_tag)     # literal markup inside the value


def _scan(text, open_re, close_tag, legal):
    """Yield (name, body) for every top-level block, honoring nesting.

    `name` is capture group 1 of open_re; a level with no name passes a pattern
    whose first group is always empty, so one implementation serves all three.

    Nesting is tried first, because that is what lets a value carry a complete
    echoed call. When nesting leaves a block unclosed - a lone, unpaired open tag
    inside a value, which is plain text - the block is rescanned without nesting,
    which reproduces the pre-boundary-scan reading. Both shapes show up in agent
    traffic and neither may cost the caller an argument.

    A block that no reading can close is DROPPED, as it was before: generation
    cut off by max_tokens must not deliver half a call.
    """
    pos = 0
    while True:
        found = _scan_once(text, pos, open_re, close_tag, legal, nest=True)
        if found is None:
            return
        name, body, next_pos, open_start, unclosed = found
        if unclosed:
            retry = _scan_once(text, open_start, open_re, close_tag, legal, nest=False)
            if retry is None or retry[4]:
                return
            name, body, next_pos = retry[0], retry[1], retry[2]
        yield name, body
        pos = next_pos


def parse_toolcalls(text: str) -> list[ToolCall]:
    # If there are outer <tool_call> wrappers, unwrap them; otherwise use the whole text
    segments: list[str] = []  # inner text of each tool-call block
    wrapped = list(_scan(text, _CALL_OPEN_SCAN, CALL_CLOSE, None))
    if wrapped:
        is_wrapped = False
        for _, inner in wrapped:
            segments.append(inner)
    else:
        # No outer wrapper — look for bare <function=...> blocks
        is_wrapped = True
        segments = [text]

    results = []
    for inner in segments:
        for func_name, func_body in _scan(inner, _FUNC_OPEN_SCAN, FUNC_CLOSE, None):
            args: dict[str, any] = {}
            for key, raw_value in _scan(
                func_body, _PARAM_OPEN_SCAN, PARAM_CLOSE, _AFTER_PARAM_CLOSE
            ):
                args[key.strip()] = coerce_param_value(raw_value.strip())

            args_json = json.dumps(args, ensure_ascii=False)
            results.append(ToolCall(function=Tool(name=func_name, arguments=args_json)))

    xlogger.debug(
        f"qwen3_coder: Parsed {len(results)} tool calls",
        {"raw_text": text, "results": results, "is_wrapped": is_wrapped},
    )
    return results
