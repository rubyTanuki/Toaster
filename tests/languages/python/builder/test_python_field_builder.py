from __future__ import annotations
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from tree_sitter import Parser
from tostr.languages.python.language import PYTHON_LANGUAGE
from tostr.graph.registry import Registry
from tostr.core.models import BaseFile, BaseField, BaseStruct
from tostr.languages.python.builders import PythonFieldBuilder


@pytest.fixture(scope="session")
def python_parser():
    return Parser(PYTHON_LANGUAGE)

@pytest.fixture
def mock_registry():
    return MagicMock(spec=Registry)

@pytest.fixture
def mock_parent_file():
    mock_file = MagicMock(spec=BaseFile, uid="app.py")
    mock_file.__class__ = BaseFile
    return mock_file

def test_python_field_builder_extracts_assignment(python_parser, mock_registry, mock_parent_file):
    python_code = "DEBUG_MODE = True"
    tree = python_parser.parse(python_code.encode("utf-8"))
    field_node = tree.root_node.children[0]
    
    builder = PythonFieldBuilder(mock_registry)
    field_obj = builder.from_node(field_node, parent=mock_parent_file)
    
    assert field_obj.name == "DEBUG_MODE"
    assert field_obj.uid == "app.py#DEBUG_MODE"
    assert field_obj.signature == "DEBUG_MODE"
