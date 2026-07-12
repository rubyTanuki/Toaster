from __future__ import annotations
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from tostr.graph.registry import Registry
from tostr.core.paths import ProjectPaths
from tostr.core.models import BaseFile
from tostr.languages.python.builders import PythonFileBuilder

@pytest.fixture
def mock_registry(tmp_path):
    registry = MagicMock(spec=Registry)
    registry.project_path = tmp_path
    registry.add_struct = MagicMock()
    registry.paths = ProjectPaths(tmp_path)
    return registry

def test_python_file_builder_extracts_imports_and_children(tmp_path, mock_registry):
    python_code = """
import os
from sys import path
from math import *

def my_function():
    pass

class MyClass:
    pass

GLOBAL_VAR = 10
"""
    path = tmp_path / "app.py"
    path.write_text(python_code)
    
    builder = PythonFileBuilder(mock_registry)
    file_obj = builder.from_path(path)
    
    # 1. Test Imports
    assert "os.py" in file_obj.imports
    assert "sys.py#path" in file_obj.imports
    assert "math.py.*" in file_obj.imports
    
    # 2. Test Package (Module path)
    assert file_obj.package == "app"
    
    # 3. Test Child Delegation
    # Should find 1 function, 1 class, 1 global variable
    assert mock_registry.add_struct.call_count == 3
    
    registered_structs = [call.args[0] for call in mock_registry.add_struct.call_args_list]
    struct_names = [s.name for s in registered_structs]
    assert "my_function" in struct_names
    assert "MyClass" in struct_names
    assert "GLOBAL_VAR" in struct_names
