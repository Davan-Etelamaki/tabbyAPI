"""
A non-streaming completion that produced no visible text must report content as an empty string,
not null.

Why this matters: the case that motivated it is a reasoning model whose whole token budget went into
reasoning_content. The caller gets HTTP 200, finish_reason="length", and message.content = null --
so a client that checks the status code and then reads content sees a successful, empty answer and
has no idea the model spent everything thinking. The text was generated; it simply never became an
answer. An empty string states that truthfully.

OpenAI's own schema reserves null content for turns that carry tool_calls, so that case is asserted
unchanged here rather than "fixed".
"""

import unittest

import common.model  # noqa: F401 - resolve import cycle ordering
from endpoints.OAI.utils.chat_completion import _compose_response


def generation(**kwargs):
    """A finish chunk as the non-streaming collector leaves it."""
    base = {
        "index": 0,
        "prompt_tokens": 41,
        "gen_tokens": 6,
        "finish_reason": "length",
    }
    base.update(kwargs)
    return base


class EmptyContentTests(unittest.TestCase):
    def test_reasoning_only_answer_reports_empty_string(self):
        resp = _compose_response(
            "req-1",
            [generation(content=None, reasoning_content="7.3 is larger because...")],
            "workstation",
            True,
        )
        message = resp.model_dump()["choices"][0]["message"]
        self.assertIn("content", message)
        self.assertEqual(message["content"], "")

    # No whitespace-only test here on purpose: deciding that content is empty is the collector's
    # job (it uses full_content.strip()), so whitespace never reaches this function as content.
    # Asserting a trim at this layer would test a rule the code does not have.

    def test_real_content_is_untouched(self):
        resp = _compose_response("req-3", [generation(content="7.3")], "workstation", True)
        self.assertEqual(resp.model_dump()["choices"][0]["message"]["content"], "7.3")

    def test_tool_call_turn_keeps_null_content(self):
        """The one place the spec allows null. Changing this would break clients that branch on it."""
        tool_call = {
            "id": "c1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"Oulu"}'},
        }
        resp = _compose_response(
            "req-4",
            [generation(content=None, tool_calls=[tool_call], finish_reason="tool_calls")],
            "workstation",
            True,
        )
        message = resp.model_dump()["choices"][0]["message"]
        self.assertIsNone(message["content"])
        self.assertIsNotNone(message["tool_calls"])

    def test_reasoning_content_still_survives(self):
        resp = _compose_response(
            "req-5", [generation(content=None, reasoning_content="thinking")], "workstation", True
        )
        message = resp.model_dump()["choices"][0]["message"]
        self.assertEqual(message["reasoning_content"], "thinking")


class UsageTests(unittest.TestCase):
    def test_non_streaming_always_carries_usage(self):
        resp = _compose_response("req-6", [generation(content="7.3")], "workstation", True)
        usage = resp.model_dump().get("usage")
        self.assertIsNotNone(usage)
        self.assertEqual(usage["prompt_tokens"], 41)
        self.assertEqual(usage["completion_tokens"], 6)

    def test_a_chunk_without_finish_reason_produces_no_usage(self):
        """Documenting the other half: usage is derived from the finish chunk, so a generation that
        never reached one has nothing to report. That is a real absence, not a swallowed value."""
        resp = _compose_response(
            "req-7", [{"index": 0, "content": "7.3", "prompt_tokens": 41}], "workstation", True
        )
        self.assertIsNone(resp.model_dump().get("usage"))


if __name__ == "__main__":
    unittest.main()