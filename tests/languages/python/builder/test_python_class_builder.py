from __future__ import annotations
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from tree_sitter import Parser
from tostr.languages.python.language import PYTHON_LANGUAGE
from tostr.graph.registry import Registry
from tostr.core.models import BaseFile, BaseClass, BaseMethod, BaseField, BaseStruct
from tostr.languages.python.builders import PythonClassBuilder


@pytest.fixture(scope="session")
def python_parser():
    """Provides a reusable Tree-sitter parser for the test session."""
    parser = Parser(PYTHON_LANGUAGE)
    return parser

@pytest.fixture
def mock_registry():
    registry = MagicMock(spec=Registry)
    registry.add_struct = MagicMock()
    return registry

@pytest.fixture
def mock_parent_file():
    """Mocks the BaseFile parent needed for UID generation."""
    mock_file = MagicMock(spec=BaseFile, uid="processor.py", path=Path("processor.py"))
    mock_file.package = "processor"
    # Tell isinstance(parent, BaseFile) to return True
    mock_file.__class__ = BaseFile 
    return mock_file

def test_python_class_builder_extracts_complex_class(python_parser, mock_registry, mock_parent_file):
    python_code = """
class DataProcessor(BaseProcessor, Serializable):
    def __init__(self, data):
        self.data = data
        
    def process(self):
        pass
        
    class InnerHelper:
        pass
"""
    
    # Parse the snippet to get the tree
    tree = python_parser.parse(python_code.encode("utf-8"))
    
    # Grab the 'class_definition' node
    class_node = None
    for child in tree.root_node.children:
        if child.type == "class_definition":
            class_node = child
            break
            
    assert class_node is not None, "Parser failed to find class_definition"

    # Instantiate the builder and process the node
    builder = PythonClassBuilder(mock_registry)
    class_obj = builder.from_node(class_node, parent=mock_parent_file)

    # 1. Test BaseStruct Properties
    assert class_obj.name == "DataProcessor"
    assert class_obj.uid == f"{mock_parent_file.uid}#DataProcessor"
    assert class_obj.parent == mock_parent_file

    # 2. Test Signature Extraction
    assert "class DataProcessor(BaseProcessor, Serializable)" in class_obj.signature

    # 3. Test Inheritance
    assert "BaseProcessor" in class_obj.inherits
    assert "Serializable" in class_obj.inherits
    assert len(class_obj.inherits) == 2

    # 4. Test Child Delegation
    # The builder should have encountered 2 methods (__init__, process), and 1 inner class
    # Field extraction in Python is a bit different, our current field builder looks for top-level assignments.
    # In this class, we have self.data = data, which is an assignment in __init__.
    # Our ClassBuilder body loop currently doesn't go deep into methods to find fields, it only looks at direct children of the class body.
    
    # Registered structs should be: __init__, process, InnerHelper
    assert mock_registry.add_struct.call_count == 3
    
    registered_structs = [call.args[0] for call in mock_registry.add_struct.call_args_list]
    struct_names = [s.name for s in registered_structs]
    assert "__init__" in struct_names
    assert "process" in struct_names
    assert "InnerHelper" in struct_names
