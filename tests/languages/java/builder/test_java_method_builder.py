from __future__ import annotations
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from tree_sitter import Parser
from tostr.languages.java.language import JAVA_LANGUAGE
from tostr.graph.registry import Registry
from tostr.core.models import BaseFile, BaseClass, BaseMethod, BaseStruct
from tostr.languages.java.builders import JavaMethodBuilder

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
        uid="com/example/OuterClass.java#OuterClass",
        name="OuterClass",
        parent=MagicMock(spec=BaseFile, uid="com/example/OuterClass.java", package="com.example"),
        imports=["java.util.*", "com.example.dep.Dependency"],
        fields=[MagicMock(name="myService", field_type="MyService")]
    )
    return mock_cls

def test_java_method_builder_extracts_complex_method(java_parser, mock_registry, mock_parent_class):
    # Snippet containing a complex method and a simple method
    java_code = """
    class Mathf {
        @Override
        public static <T> List<T> processData(int count, String name) {
            return new ArrayList<>();
        }
        
        void ping() {}
    }
    """
    
    tree = java_parser.parse(java_code.encode("utf-8"))
    
    # 1. Find the method nodes inside the class body
    method_nodes = []
    class_node = tree.root_node.children[0]
    body_node = class_node.child_by_field_name("body")
    
    for child in body_node.children:
        if child.type == "method_declaration":
            method_nodes.append(child)
            
    assert len(method_nodes) == 2, f"Parser failed to find 2 methods, found: {len(method_nodes)}"

    builder = JavaMethodBuilder(mock_registry)
    
    # --- TEST 1: The Complex Method ---
    complex_method_node = method_nodes[0]
    method_obj = builder.from_node(complex_method_node, parent=mock_parent_class)

    # Core Properties
    assert method_obj.name == "processData", f"Expected \'processData\', got {method_obj.name}"
    assert method_obj.parent == mock_parent_class
    
    # Signature Tests
    assert "@Override" in method_obj.signature
    assert "public" in method_obj.signature
    assert "static" in method_obj.signature
    assert "<T>" in method_obj.signature
    assert "List<T>" in method_obj.signature
    assert "processData(int, String)" in method_obj.signature
    
    # Arity & Parameters
    assert method_obj.arity == 2, f"Expected arity 2, got {method_obj.arity}"
    
    # UID Test (Crucial for method overloading support)
    expected_uid = "com/example/OuterClass.java#OuterClass.processData(int, String)"
    assert method_obj.uid == expected_uid, f"Expected {expected_uid}, got {method_obj.uid}"


    # --- TEST 2: The Simple Method ---
    simple_method_node = method_nodes[1]
    simple_obj = builder.from_node(simple_method_node, parent=mock_parent_class)
    
    assert simple_obj.name == "ping"
    assert simple_obj.arity == 0
    assert "void ping()" in simple_obj.signature
    
    # Check empty parameter UID
    expected_simple_uid = "com/example/OuterClass.java#OuterClass.ping()"
    assert simple_obj.uid == expected_simple_uid, f"Expected {expected_simple_uid}, got {simple_obj.uid}"
