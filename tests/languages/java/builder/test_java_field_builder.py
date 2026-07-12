from __future__ import annotations
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from tree_sitter import Parser
from tostr.languages.java.language import JAVA_LANGUAGE
from tostr.graph.registry import Registry
from tostr.core.models import BaseFile, BaseClass, BaseField, BaseStruct
from tostr.languages.java.builders import JavaFieldBuilder

@pytest.fixture(scope="session")
def java_parser():
    """Provides a reusable Tree-sitter parser for the test session."""
    parser = Parser(JAVA_LANGUAGE)
    return parser

@pytest.fixture
def mock_registry():
    registry = MagicMock(spec=Registry)
    return registry

@pytest.fixture
def mock_parent_class():
    """Mocks the BaseClass parent needed for UID generation."""
    mock_cls = MagicMock(
        spec=BaseClass,
        uid="com/example/TestClass.java#TestClass",
        name="TestClass",
        parent=MagicMock(spec=BaseFile, uid="com/example/TestClass.java", package="com.example")
    )
    return mock_cls

def test_java_field_builder_extracts_fields(java_parser, mock_registry, mock_parent_class):
    # Snippet containing a complex field with a comment, and a simple generic field
    java_code = """
    class Constants {
        @Serialized // This comment should be ignored
        public static final double TAU = 6.28;
        
        private List<String> activeUsers;
    }
    """
    
    tree = java_parser.parse(java_code.encode("utf-8"))
    
    # Find the field nodes inside the class body
    field_nodes = []
    class_node = tree.root_node.children[0]
    body_node = class_node.child_by_field_name("body")
    
    for child in body_node.children:
        if child.type == "field_declaration":
            field_nodes.append(child)
            
    assert len(field_nodes) == 2, f"Parser failed to find 2 fields, found: {len(field_nodes)}"

    builder = JavaFieldBuilder(mock_registry)
    
    # --- TEST 1: The Complex Field ---
    complex_field_node = field_nodes[0]
    tau_field = builder.from_node(complex_field_node, parent=mock_parent_class)

    # Core Properties
    assert tau_field.name == "TAU"
    assert tau_field.field_type == "double"
    assert tau_field.parent == mock_parent_class
    
    # Signature Tests (Ensuring the comment was skipped and order is correct)
    expected_tau_sig = "@Serialized public static final double TAU"
    assert tau_field.signature == expected_tau_sig, f"Expected \'{expected_tau_sig}\\' , got \'{tau_field.signature}\'"
    assert "This comment should be ignored" not in tau_field.signature
    
    # UID Test (Ensuring NO type information is appended to fields)
    expected_tau_uid = "com/example/TestClass.java#TestClass.TAU"
    assert tau_field.uid == expected_tau_uid, f"Expected \'{expected_tau_uid}\\' , got \'{tau_field.uid}\'"


    # --- TEST 2: The Generic Field ---
    generic_field_node = field_nodes[1]
    users_field = builder.from_node(generic_field_node, parent=mock_parent_class)
    
    assert users_field.name == "activeUsers"
    assert users_field.field_type == "List<String>"
    
    expected_users_sig = "private List<String> activeUsers"
    assert users_field.signature == expected_users_sig
    
    expected_users_uid = "com/example/TestClass.java#TestClass.activeUsers"
    assert users_field.uid == expected_users_uid
