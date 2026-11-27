import ast
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SITE = REPO / "endpoints/OAI/utils/chat_completion.py"


class ConstructionWiringTests(unittest.TestCase):
    """
    The ghost-name defect (undeclared-but-structured tool names reaching the
    client) stays closed only while the runtime constructs the delta streamer
    WITH the request's declared names. This fence parses the real
    construction site, so a refactor that drops the argument goes RED
    instead of reintroducing the ghost silently.
    """

    def test_streamer_constructed_with_declared_names(self):
        tree = ast.parse(SITE.read_text())
        calls = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and getattr(n.func, "id", getattr(n.func, "attr", None))
            == "QwenToolCallDeltaStreamer"
        ]
        self.assertEqual(len(calls), 1, "expected exactly one construction site")
        call = calls[0]
        kwargs = {kw.arg for kw in call.keywords}
        passes_names = "valid_names" in kwargs or any(
            isinstance(a, ast.Call)
            and getattr(a.func, "id", getattr(a.func, "attr", None))
            == "tool_names_from_tools"
            for a in call.args[1:]
        )
        self.assertTrue(
            passes_names,
            "construction site lost the declared-names argument: ghost tool "
            "names stream to clients again (valid_names must be populated "
            "from params.tools)",
        )

    def test_helper_reads_both_tool_shapes(self):
        from endpoints.OAI.utils.toolcall_formats.common import (
            tool_names_from_tools,
        )

        obj = type("T", (), {"function": type("F", (), {"name": "get_weather"})()})()
        self.assertEqual(tool_names_from_tools([obj]), ["get_weather"])
        self.assertEqual(
            tool_names_from_tools([{"type": "function", "function": {"name": "declared_ok"}}]),
            ["declared_ok"],
        )
        self.assertIsNone(tool_names_from_tools(None))
        self.assertIsNone(tool_names_from_tools([]))


if __name__ == "__main__":
    unittest.main()
