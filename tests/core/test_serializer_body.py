from __future__ import annotations
from pathlib import Path

from tostr.serializer import tost
from tostr.core.models import BaseFile, Directory


def test_inspect_includes_file_body_when_requested():
    f = BaseFile(name="index.html", uid="index.html", path=Path("index.html"))
    f.body = "<html><body><h1>Hi</h1></body></html>"

    with_body = tost.dump(f, include_body=True)
    without_body = tost.dump(f, include_body=False)

    assert with_body.body == "<html><body><h1>Hi</h1></body></html>"
    assert without_body.body is None


def test_inspect_directory_has_no_body_even_when_requested():
    d = Directory(path=Path("."), uid=".")
    # Directories carry no body; requesting one must not fabricate or error.
    assert tost.dump(d, include_body=True).body is None
