from __future__ import annotations
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from tree_sitter import Parser
from tostr.languages.python.language import PYTHON_LANGUAGE
from tostr.graph.registry import Registry
from tostr.core.models import BaseFile, BaseClass, BaseMethod, BaseStruct
from tostr.languages.python.builders import PythonMethodBuilder


@pytest.fixture(scope="session")
def python_parser():
    return Parser(PYTHON_LANGUAGE)

@pytest.fixture
def mock_registry():
    registry = MagicMock(spec=Registry)
    return registry

@pytest.fixture
def mock_parent_class():
    mock_class = MagicMock(spec=BaseClass, uid="app.MyClass")
    mock_class.name = "MyClass"
    mock_class.__class__ = BaseClass
    return mock_class

def test_python_method_builder_extracts_function_details(python_parser, mock_registry, mock_parent_class):
    python_code = """
def calculate_total(self, items, tax=0.1):
    result = self.do_math(items)
    return result + tax
"""
    tree = python_parser.parse(python_code.encode("utf-8"))
    method_node = tree.root_node.children[0]
    
    builder = PythonMethodBuilder(mock_registry)
    method_obj = builder.from_node(method_node, parent=mock_parent_class)
    
    assert method_obj.name == "calculate_total"
    assert method_obj.arity == 2  # self excluded; items + tax
    assert "def calculate_total(self, items, tax=0.1)" in method_obj.signature
    
    # Check dependencies
    # self.do_math(items) should be detected
    assert ("do_math", 1, "self", False) in method_obj.dependency_names
