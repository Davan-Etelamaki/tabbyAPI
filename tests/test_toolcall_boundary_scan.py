"""Boundary-scan tests for the qwen3_coder parser (tabbyAPI-09).

Tag literals are assembled from chr() so this file, and any transport carrying it
inside a tool-call payload, never contains raw call markup.

WHY these cases exist: the parser used to take the FIRST close tag of every block,
so an argument value that had to carry literal markup ended there and the tail of
the value plus every later parameter vanished silently. The shapes below are the
ones counted in Launchers/research/litellm-tool-call-integrity (F1) and the ones
this station's own agents hit daily (writing a file, a shell command or a prompt
that quotes this wire format).
"""

import json
import unittest

from endpoints.OAI.utils.toolcall_formats import qwen3_coder
from endpoints.OAI.utils.toolcall_formats.common import coerce_param_value

LT, GT = chr(60), chr(62)


def param_open(name):
    return LT + "parameter=" + name + GT


def param_close():
    return LT + "/parameter" + GT


def func_open(name):
    return LT + "function=" + name + GT


def func_close():
    return LT + "/function" + GT


def wrap(inner):
    return LT + "tool_call>" + inner + LT + "/tool_call" + GT


def call(name, params, wrapped=True):
    body = func_open(name) + "\n"
    for key, value in params:
        body += param_open(key) + "\n" + value + "\n" + param_close() + "\n"
    body += func_close()
    return wrap(body) if wrapped else body


def parsed(text):
    return [(c.function.name, json.loads(c.function.arguments)) for c in qwen3_coder.parse_toolcalls(text)]


class MarkupInsideAValueTests(unittest.TestCase):
    def test_close_tag_inside_a_value_keeps_the_tail_and_the_next_argument(self):
        # the census shape: prose continues after a literal parameter close
        text = call("write_file", [("content", "A " + param_close() + " B"), ("purpose", "why")])
        self.assertEqual(parsed(text), [("write_file", {"content": "A " + param_close() + " B",
                                                        "purpose": "why"})])

    def test_lone_open_tag_inside_a_value_stays_plain_text(self):
        # no matching close for the inner open: nesting must not swallow the block
        literal = param_open("y")
        text = call("a", [("x", "text " + literal + " more"), ("z", "kept")])
        self.assertEqual(parsed(text), [("a", {"x": "text " + literal + " more", "z": "kept"})])

    def test_echoed_inner_call_does_not_become_a_second_call(self):
        inner = call("read", [("file_path", "x.py")], wrapped=False)
        text = call("bash", [("command", "prefix " + inner + " suffix"), ("description", "echo")])
        got = parsed(text)
        self.assertEqual(len(got), 1, got)
        self.assertEqual(got[0][0], "bash")
        self.assertIn(param_open("file_path"), got[0][1]["command"])
        self.assertEqual(got[0][1]["description"], "echo")

    def test_unbalanced_echo_falls_back_without_crashing(self):
        # an echo that itself got cut leaves an open tag with no close: nesting
        # cannot pair it, so the scan falls back to the plain reading. The call
        # still delivers its first argument instead of vanishing.
        broken = func_open("read") + "\n" + param_open("file_path") + "\nx.py"
        text = call("bash", [("command", "saw " + broken), ("description", "echo")])
        got = parsed(text)
        self.assertEqual(len(got), 1, got)
        self.assertEqual(got[0][0], "bash")
        self.assertIn("saw ", got[0][1]["command"])

    def test_function_close_inside_a_value_does_not_split_the_call(self):
        inner = func_open("tools") + "\nbody\n" + func_close()
        text = call("read", [("file_path", "a.py"), ("note", "saw " + inner), ("mode", "r")])
        got = parsed(text)
        self.assertEqual(len(got), 1, got)
        self.assertEqual(got[0][1]["file_path"], "a.py")
        self.assertEqual(got[0][1]["mode"], "r")
        self.assertIn(func_open("tools"), got[0][1]["note"])


class WellFormedCallsUnchangedTests(unittest.TestCase):
    def test_single_call(self):
        self.assertEqual(parsed(call("run", [("cmd", "ls -l")])),
                         [("run", {"cmd": "ls -l"})])

    def test_two_calls_in_one_block(self):
        text = wrap(call("a", [("x", "1")], wrapped=False) + "\n"
                    + call("b", [("y", "2")], wrapped=False))
        self.assertEqual(parsed(text), [("a", {"x": 1}), ("b", {"y": 2})])

    def test_prose_around_the_call_inside_the_wrapper(self):
        text = wrap("prose " + call("a", [("x", "1")], wrapped=False) + " trailing prose")
        self.assertEqual(parsed(text), [("a", {"x": 1})])

    def test_coercion_is_untouched(self):
        text = call("m", [("n", "7"), ("b", "true"), ("o", '{"k": [1, 2]}'),
                          ("s", "café"), ("z", "null"), ("e", "")])
        self.assertEqual(parsed(text),
                         [("m", {"n": 7, "b": True, "o": {"k": [1, 2]},
                                 "s": "café", "z": None, "e": ""})])

    def test_duplicate_parameter_keys_keep_the_last(self):
        inner = (func_open("a") + "\n" + param_open("x") + "\n1\n" + param_close() + "\n"
                 + param_open("x") + "\n2\n" + param_close() + "\n" + func_close())
        self.assertEqual(parsed(wrap(inner)), [("a", {"x": 2})])


class TruncatedGenerationTests(unittest.TestCase):
    def test_unclosed_call_is_dropped_not_delivered_half(self):
        # max_tokens cut: a partial call must never reach the client
        text = wrap(call("a", [("x", "1")], wrapped=False)[: -len(func_close())])
        self.assertEqual(qwen3_coder.parse_toolcalls(text), [])

    def test_unclosed_value_is_dropped(self):
        text = wrap(func_open("a") + "\n" + param_open("x") + "\nhalf a value")
        self.assertEqual(qwen3_coder.parse_toolcalls(text), [])


class CoercionContractTests(unittest.TestCase):
    def test_value_still_goes_through_coerce_param_value(self):
        raw = "A " + param_close() + " B"
        self.assertEqual(coerce_param_value(raw), raw)


if __name__ == "__main__":
    unittest.main()