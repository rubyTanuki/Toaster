from __future__ import annotations
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from tree_sitter import Parser
from tostr.languages.java.language import JAVA_LANGUAGE
from tostr.graph.registry import Registry
from tostr.core.models import BaseFile, BaseClass, BaseMethod, BaseField, BaseStruct
from tostr.languages.java.builders import JavaClassBuilder


@pytest.fixture(scope="session")
def java_parser():
    """Provides a reusable Tree-sitter parser for the test session."""
    parser = Parser(JAVA_LANGUAGE)
    return parser

@pytest.fixture
def mock_registry():
    registry = MagicMock(spec=Registry)
    registry.add_struct = MagicMock()
    return registry

@pytest.fixture
def mock_parent_file():
    """Mocks the BaseFile parent needed for UID generation."""
    mock_file = MagicMock(spec=BaseFile, uid="com/tostr/DataProcessor.java", path=Path("src/main/java/com/tostr/DataProcessor.java"))
    mock_file.package = "com.tostr"
    # Tell isinstance(parent, BaseFile) to return True
    mock_file.__class__ = BaseFile 
    return mock_file

def test_java_class_builder_extracts_complex_class(java_parser, mock_registry, mock_parent_file):
    # This snippet covers annotations, generics, extends, implements, and inner children
    java_code = """
    @Component
    public abstract class DataProcessor<T> extends BaseProcessor implements Serializable, Runnable {
        
        private T data;
        
        public void process() { }
        
        class InnerHelper {}
    }
    """
    
    # Parse the snippet to get the tree
    tree = java_parser.parse(java_code.encode("utf-8"))
    
    # Grab the 'class_declaration' node
    class_node = None
    for child in tree.root_node.children:
        if child.type == "class_declaration":
            class_node = child
            break
            
    assert class_node is not None, "Parser failed to find class_declaration"

    # Instantiate the builder and process the node
    builder = JavaClassBuilder(mock_registry)
    
    # If your BaseStruct models don't mock add_child well, 
    # you might need to patch BaseClass.add_child or ensure the real method works.
    class_obj = builder.from_node(class_node, parent=mock_parent_file)

    # 1. Test BaseStruct Properties
    assert class_obj.name == "DataProcessor"
    assert class_obj.uid == f"{mock_parent_file.uid}#DataProcessor"
    assert class_obj.parent == mock_parent_file

    # 2. Test Signature Extraction
    assert "@Component" in class_obj.signature
    assert "public" in class_obj.signature
    assert "abstract" in class_obj.signature
    assert "class DataProcessor<T>" in class_obj.signature

    # 3. Test Inheritance (Checking the named_children fixes)
    assert "BaseProcessor" in class_obj.inherits
    assert "Serializable" in class_obj.inherits
    assert "Runnable" in class_obj.inherits
    assert len(class_obj.inherits) == 3, f"Expected 3 inherits, but got {class_obj.inherits}"

    # 4. Test Child Delegation
    # The builder should have encountered 1 field, 1 method, and 1 inner class
    assert mock_registry.add_struct.call_count == 3
    
    # Verify the correct types were sent to the registry
    registered_structs = [call.args[0] for call in mock_registry.add_struct.call_args_list]
    
    struct_names = [s.name for s in registered_structs]
    assert "data" in struct_names # The field
    assert "process" in struct_names # The method
    assert "InnerHelper" in struct_names # The inner class
