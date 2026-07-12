from __future__ import annotations
from pathlib import Path
from unittest.mock import MagicMock

from tostr.graph.registry import Registry
from tostr.core.paths import ProjectPaths
from tostr.core.models import BaseFile, Directory
from tostr.core.builders import BaseFileBuilder
from tostr.serializer import tost
from tostr.languages.python.builders import PythonFileBuilder


def _mock_registry(tmp_path):
    registry = MagicMock(spec=Registry)
    registry.project_path = tmp_path
    registry.add_struct = MagicMock()
    registry.paths = ProjectPaths(tmp_path)
    return registry


def test_file_to_dict_serializes_line_bounds():
    # Line bounds are 1-indexed source lines (a single-line file would be 1..1).
    f = BaseFile(name="page.html", uid="page.html", path=Path("page.html"))
    f.start_line = 1
    f.end_line = 42

    d = f.to_dict()
    assert d["start_line"] == 1
    assert d["end_line"] == 42


def test_file_from_dict_hydrates_line_bounds(tmp_path):
    # from_dict consumes a DB-shaped row: dependency-string columns arrive as JSON
    # strings (the cache json-serializes them on the way in), so mirror that here.
    row = {
        "uid": "page.html",
        "name": "page.html",
        "path": "page.html",
        "start_line": 1,
        "end_line": 42,
        "inbound_dependency_strings": "[]",
        "outbound_dependency_strings": "[]",
    }
    hydrated = BaseFileBuilder(_mock_registry(tmp_path)).from_dict(row)
    assert hydrated.start_line == 1
    assert hydrated.end_line == 42


def test_inspect_result_exposes_file_line_bounds():
    f = BaseFile(name="page.html", uid="page.html", path=Path("page.html"))
    f.start_line = 1
    f.end_line = 42

    result = tost.dump(f)
    assert result.start_line == 1
    assert result.end_line == 42


def test_parse_produces_1_indexed_clamped_bounds(tmp_path):
    # The real parse path must emit 1-indexed lines that match an editor/traceback, and
    # end_line must clamp to the last line of actual content (not overrun the trailing
    # newline). This 5-line file ends at L5, not L6.
    code = "import os\n\n\ndef my_func():\n    return os.getcwd()\n"
    #       ^L1                ^L4         ^L5 (last content line)
    path = tmp_path / "app.py"
    path.write_bytes(code.encode())

    f = PythonFileBuilder(_mock_registry(tmp_path)).from_path(path)

    # File spans L1-L5 — start at 1 (not 0), end at the last content line (not 6).
    assert (f.start_line, f.end_line) == (1, 5)

    func = next(s for s in f.all_children if s.name == "my_func")
    # The def is on L4 and its last content line (the return) is L5.
    assert (func.start_line, func.end_line) == (4, 5)


def test_directory_has_no_line_bounds():
    # Directories intentionally omit line bounds; dump must not fabricate them.
    d = Directory(path=Path("."), uid=".")
    assert not hasattr(d, "start_line")
    # The serializer's getattr fallback keeps directories at the 0 default without error.
    assert tost.dump(d).start_line == 0
