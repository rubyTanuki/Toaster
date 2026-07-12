from __future__ import annotations
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from tostr.graph.registry import Registry
from tostr.core.paths import ProjectPaths
from tostr.core.models import BaseFile
from tostr.core.providers import LanguageProvider
from tostr.languages.html.builders import HtmlBuilder, HtmlFileBuilder


@pytest.fixture
def mock_registry(tmp_path):
    registry = MagicMock(spec=Registry)
    registry.project_path = tmp_path
    registry.add_struct = MagicMock()
    registry.paths = ProjectPaths(tmp_path)
    return registry


def test_html_extension_routing():
    # Both common extensions resolve to the html language.
    assert LanguageProvider.language_for_extension(".html") == "html"
    assert LanguageProvider.language_for_extension(".htm") == "html"


def test_html_uses_noop_resolver(mock_registry):
    # HTML has no dependency resolution; it falls back to the base (no-op) resolver.
    mock_registry.language = "auto"
    from tostr.graph.resolver import BaseDependencyResolver
    resolver = LanguageProvider.get_resolver(mock_registry, ".html")
    assert isinstance(resolver, BaseDependencyResolver)


def test_html_builder_builds_file_only(mock_registry, tmp_path):
    html = """<!DOCTYPE html>
<html>
<head><title>Test Page</title></head>
<body>
  <h1>Hello</h1>
  <p>A small page for parsing.</p>
</body>
</html>
"""
    path = tmp_path / "index.html"
    path.write_text(html)

    file_obj = HtmlBuilder(mock_registry).build_file().from_path(path)

    # 1. A BaseFile is produced with body + diff_hash populated.
    assert isinstance(file_obj, BaseFile)
    assert file_obj.body == html
    assert file_obj.diff_hash

    # 2. The document is parsed so the file carries a tree-sitter node.
    assert file_obj.node is not None
    assert file_obj.node.type == "document"

    # 3. File-level only: no child structs are extracted or registered.
    assert mock_registry.add_struct.call_count == 0
    assert all(len(s) == 0 for s in file_obj.children.values())


def test_html_builder_factory_returns_file_builder(mock_registry):
    assert isinstance(HtmlBuilder(mock_registry).build_file(), HtmlFileBuilder)
