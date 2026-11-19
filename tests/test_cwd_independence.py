"""Runtime paths must not depend on the process working directory.

Measured in production 2026-11-17: the checkout a running engine was started from
was deleted, which unlinked the process working directory. `os.getcwd()` then
raises FileNotFoundError for the life of the process, so every relative runtime
path -- the draft-model directory, the log sink, the templates directory --
raises with it. `GET /v1/models` answered 500 on every poll and the log file
became an unlinked inode.
"""

import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import textwrap

import pytest

APP_ROOT = pathlib.Path(__file__).resolve().parent.parent

MODEL_CONFIG = json.dumps({"architectures": ["Qwen2ForCausalLM"]})

IMPORT_MODEL_LIST = """
    import pathlib

    from endpoints.core.utils.model import get_model_list
    """


def run_with_unlinked_cwd(setup: str, call: str) -> subprocess.CompletedProcess:
    """Import `setup`, then delete the working directory, then run `call`.

    Production imported torch while the checkout still existed, so a
    `setup`/`call` split is the faithful model: the tree vanished between boot
    and request. Importing MKL after the deletion aborts for its own reasons and
    would hide the path bug. The child unlinks the directory itself, because a
    parent cannot `chdir` into a directory that no longer exists.
    """

    with tempfile.TemporaryDirectory() as tmp:
        gone = pathlib.Path(tmp) / "checkout"
        gone.mkdir()
        unlink = (
            "import os, shutil\n"
            f"os.chdir({str(gone)!r})\n"
            f"shutil.rmtree({str(gone)!r})\n"
            "try:\n"
            "    os.getcwd()\n"
            "except FileNotFoundError:\n"
            "    print('__PRE__ True')\n"
            "else:\n"
            "    print('__PRE__ False')\n"
        )
        # -u: the precondition line must survive an abort in a later call.
        # Inherit the environment: torch's bundled libraries need the paths the
        # venv relies on, and a stripped env aborts before it can be measured.
        return subprocess.run(
            [
                sys.executable,
                "-u",
                "-c",
                textwrap.dedent(setup).strip()
                + "\n"
                + unlink
                + textwrap.dedent(call).strip(),
            ],
            cwd=gone,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(APP_ROOT)},
        )


def payload(result: subprocess.CompletedProcess) -> str:
    """Fail unless the unlinked-cwd precondition landed and the call did not raise."""

    lines = result.stdout.splitlines()
    assert lines and lines[0] == "__PRE__ True", (
        "the child's working directory was not unlinked, so this test measured "
        f"nothing.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "FileNotFoundError" not in result.stderr, result.stderr
    assert result.returncode == 0, result.stderr
    return "\n".join(lines[1:])


def test_get_model_list_survives_an_unlinked_cwd():
    """The production failure: resolving the draft directory with no cwd.

    Before the anchor, `pathlib.Path("models").resolve()` called `getcwd()` here
    and `GET /v1/models` answered 500. Afterwards a relative draft directory
    means the application tree, so the request still answers and advertises the
    model directories it was pointed at.
    """

    with tempfile.TemporaryDirectory() as tmp:
        models_dir = pathlib.Path(tmp) / "models"
        draft_dir = models_dir / "draft"
        draft_dir.mkdir(parents=True)
        (draft_dir / "config.json").write_text(MODEL_CONFIG)

        result = run_with_unlinked_cwd(
            IMPORT_MODEL_LIST,
            f"""
            listing = get_model_list(pathlib.Path({str(models_dir)!r}), "models")
            print(listing.model_dump_json())
            """,
        )

    advertised = json.loads(payload(result))["data"]
    assert len(advertised) == 1, advertised
    assert "draft" in json.dumps(advertised), advertised


def test_absent_draft_directory_lists_no_models():
    """Positive control for the empty case: an empty tree lists nothing, no error."""

    with tempfile.TemporaryDirectory() as tmp:
        models_dir = pathlib.Path(tmp) / "models"
        models_dir.mkdir()

        result = run_with_unlinked_cwd(
            IMPORT_MODEL_LIST,
            f"""
            print(len(get_model_list(pathlib.Path({str(models_dir)!r}), "models").data))
            """,
        )

    assert payload(result).strip() == "0"


def test_log_sink_and_templates_are_anchored_to_the_app():
    """Neither the log sink nor the templates directory may come from the cwd."""

    result = run_with_unlinked_cwd(
        """
        import loguru

        recorded = []
        real_add = loguru.logger.add

        def spy(sink, *args, **kwargs):
            recorded.append(str(sink))
            return real_add(sink, *args, **kwargs)

        loguru.logger.add = spy

        from common import logger
        """,
        """
        import json
        import pathlib

        logger.setup_logger()

        from common.paths import LOGS_DIR, TEMPLATES_DIR
        from common.templating import get_all_templates

        print(
            json.dumps(
                {
                    "logs_dir": str(LOGS_DIR),
                    "templates_dir": str(TEMPLATES_DIR),
                    "sinks": [s for s in recorded if s.endswith(".log")],
                    "templates": sorted(p.name for p in get_all_templates()),
                }
            )
        )
        """,
    )

    observed = json.loads(payload(result))
    assert pathlib.Path(observed["logs_dir"]) == APP_ROOT / "logs"
    assert pathlib.Path(observed["templates_dir"]) == APP_ROOT / "templates"
    assert observed["templates"], "no prompt templates found"
    assert observed["sinks"], "no file sink registered"

    for sink in observed["sinks"]:
        assert pathlib.Path(sink).is_absolute(), sink
        assert pathlib.Path(sink).parent == APP_ROOT / "logs", sink


@pytest.mark.parametrize("relative", ["models", "loras", "logs/debug"])
def test_relative_config_paths_anchor_to_the_app(relative):
    """A configured relative path means "inside the app", not "wherever we started"."""

    from common.paths import APP_ROOT, runtime_path

    assert runtime_path(relative) == APP_ROOT / relative
    assert runtime_path("/srv/models") == pathlib.Path("/srv/models")


def test_no_relative_path_literal_remains_in_served_code():
    """Guards the class, not one instance: a relative literal is how the bug returns.

    `resolve()` and `abspath()` call `os.getcwd()`, and a relative `open()`/`glob()`
    resolves against the working directory too, so any literal relative path in
    served code reintroduces the 2026-11-17 outage. The names in `common/paths.py`
    are the only sanctioned roots.
    """

    served = [
        APP_ROOT / "common",
        APP_ROOT / "endpoints",
        APP_ROOT / "backends",
        APP_ROOT / "main.py",
    ]
    offenders = []

    for root in served:
        paths = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for path in paths:
            if path == APP_ROOT / "common" / "paths.py":
                continue
            for lineno, line in enumerate(path.read_text().splitlines(), start=1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                for match in re.finditer(r"""Path\(\s*["']([^"']*)["']""", stripped):
                    if not match.group(1).startswith(("/", "\\")):
                        offenders.append(
                            f"{path.relative_to(APP_ROOT)}:{lineno}: {stripped}"
                        )

    assert not offenders, "relative path literal(s):\n" + "\n".join(offenders)