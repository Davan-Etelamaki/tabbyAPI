"""Runtime paths, anchored to the application tree.

A relative path resolved against the process working directory breaks when that
directory moves or is deleted while the server is serving: `Path.resolve()` and
`os.path.abspath()` call `os.getcwd()`, which raises FileNotFoundError for the
life of a process whose working directory has been unlinked. Production hit this
2026-11-17, when the checkout a running engine was started from was removed;
`GET /v1/models` answered 500 on every poll and its log file became an unlinked
inode.
"""

import pathlib

APP_ROOT = pathlib.Path(__file__).resolve().parent.parent

TEMPLATES_DIR = APP_ROOT / "templates"
LOGS_DIR = APP_ROOT / "logs"
SAMPLER_OVERRIDES_DIR = APP_ROOT / "sampler_overrides"


def runtime_path(path) -> pathlib.Path:
    """Interpret a configured path: absolute stays, relative means inside the app."""

    path = pathlib.Path(path)
    return path if path.is_absolute() else APP_ROOT / path