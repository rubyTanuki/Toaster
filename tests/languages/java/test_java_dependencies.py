from __future__ import annotations
import pytest
from tostr.graph.registry import Registry
from tostr.core.paths import ProjectPaths
from tostr.languages.java.builders import JavaFileBuilder
from tostr.core.models import BaseStruct

@pytest.fixture
def registry(tmp_path):
    return Registry(ProjectPaths(tmp_path))

def test_java_dependency_parsing(tmp_path, registry):
    """Tests that JavaMethodBuilder correctly identifies method calls."""
    java_code = """
    package com.example;
    public class Test {
        public void method1() {
            method2();
            method3(1, 2);
        }
        public void method2() {}
        public void method3(int a, int b) {}
    }
    """
    path = tmp_path / "Test.java"
    path.write_text(java_code)
    
    builder = JavaFileBuilder(registry)
    file_obj = builder.from_path(path)
    registry.add_struct(file_obj)

    method1 = [m for m in registry.methods if m.name == "method1"][0]
    
    # Verify dependency names and arities are parsed
    assert ("method2", 0, None, False) in method1.dependency_names
    assert ("method3", 2, None, False) in method1.dependency_names

def test_java_dependency_resolution_local(tmp_path, registry):
    """Tests resolution of local method calls (same class)."""
    java_code = """
    package com.example;
    public class Test {
        public void method1() {
            method2();
        }
        public void method2() {}
    }
    """
    path = tmp_path / "Test.java"
    path.write_text(java_code)
    
    builder = JavaFileBuilder(registry)
    file_obj = builder.from_path(path)
    registry.add_struct(file_obj)

    # Trigger resolution
    file_obj.resolve_dependencies()
    
    method1 = [m for m in registry.methods if m.name == "method1"][0]
    method2 = [m for m in registry.methods if m.name == "method2"][0]
    
    # Verify that method1 now has an outbound dependency on method2
    assert method2 in method1.outbound_dependencies

def test_java_dependency_resolution_imported(tmp_path, registry):
    """Tests resolution of method calls from imported classes."""
    java_a = """
    package com.example;
    public class A {
        public void methodA() {}
    }
    """
    java_b = """
    package com.example;
    import com.example.A;
    public class B {
        public void methodB() {
            A a = new A();
            a.methodA();
        }
    }
    """
    
    (tmp_path / "com/example").mkdir(parents=True)
    path_a = tmp_path / "com/example/A.java"
    path_a.write_text(java_a)
    path_b = tmp_path / "com/example/B.java"
    path_b.write_text(java_b)
    
    builder = JavaFileBuilder(registry)
    
    for p in [path_a, path_b]:
        file_obj = builder.from_path(p)
        registry.add_struct(file_obj)

    # Resolve all
    for file in registry.files:
        file.resolve_dependencies()
            
    method_b = [m for m in registry.methods if m.name == "methodB"][0]
    method_a = [m for m in registry.methods if m.name == "methodA"][0]
    
    # This tests the "IMPORTED" logic in BaseMethod.resolve_dependencies
    assert method_a.parent in method_b.outbound_dependencies
