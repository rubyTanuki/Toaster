from __future__ import annotations
import pytest
from pathlib import Path
from tostr.graph.registry import Registry
from tostr.core.paths import ProjectPaths
from tostr.core.models import BaseFile, BaseClass, BaseMethod, BaseField

@pytest.fixture
def registry(tmp_path):
    return Registry(ProjectPaths(tmp_path))

def test_agnostic_local_resolution(registry):
    """Tests that the resolver can find a local function in a file (no class)."""
    file_obj = BaseFile(uid="test.py", name="test.py", registry=registry)
    
    # Define two functions at file level
    func1 = BaseMethod(uid="test.py#func1", name="func1", arity=0, parent=file_obj)
    func2 = BaseMethod(uid="test.py#func2", name="func2", arity=0, parent=file_obj)
    
    # func1 calls func2
    func1.dependency_names = [("func2", 0, None, False)]
    
    file_obj.add_child(func1)
    file_obj.add_child(func2)
    registry.add_struct(file_obj)
    registry.add_struct(func1)
    registry.add_struct(func2)
    
    # Resolve
    resolver = registry.get_resolver()
    resolver.resolve_method_dependencies(func1)
    
    assert func2 in func1.outbound_dependencies

def test_agnostic_receiver_heuristic(registry):
    """Tests that the resolver uses receiver types from parent fields (even in files)."""
    file_obj = BaseFile(uid="test.py", name="test.py", registry=registry)
    
    # Global variable 'logger' of type 'Logger'
    field = BaseField(uid="test.py#logger", name="logger", field_type="Logger", parent=file_obj)
    
    # A class 'Logger' with method 'log'
    logger_class = BaseClass(uid="Logger", name="Logger", registry=registry)
    log_method = BaseMethod(uid="Logger#log", name="log", arity=1, parent=logger_class)
    logger_class.add_child(log_method)
    
    # Function calling logger.log("msg")
    func = BaseMethod(uid="test.py#main", name="main", arity=0, parent=file_obj)
    func.dependency_names = [("log", 1, "logger", False)]
    
    file_obj.add_child(field)
    file_obj.add_child(func)
    
    registry.add_struct(file_obj)
    registry.add_struct(field)
    registry.add_struct(logger_class)
    registry.add_struct(log_method)
    registry.add_struct(func)
    
    # Resolve
    resolver = registry.get_resolver()
    resolver.resolve_method_dependencies(func)
    
    assert log_method in func.outbound_dependencies

def test_agnostic_import_resolution(registry):
    """Tests that the resolver respects normalized imports."""
    file_obj = BaseFile(uid="test.py", name="test.py", registry=registry)
    # Normalized import candidate: path-format member UID.
    file_obj.imports = ["other.py#Member"]

    other_member = BaseClass(uid="other.py#Member", name="Member", registry=registry)
    registry.add_struct(other_member)
    
    # Code creating 'Member'
    func = BaseMethod(uid="test.py#main", name="main", arity=0, parent=file_obj)
    func.dependency_names = [("Member", 0, None, True)] # is_creation=True
    
    file_obj.add_child(func)
    registry.add_struct(file_obj)
    registry.add_struct(func)
    
    # Resolve
    resolver = registry.get_resolver()
    resolver.resolve_method_dependencies(func)
    
    assert other_member in func.outbound_dependencies
