from __future__ import annotations
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from tostr.core.registry import Registry
from tostr.languages.java.builders import JavaFileBuilder
from tostr.core.models import BaseStruct

@pytest.fixture
def mock_registry():
    registry = MagicMock(spec=Registry)
    registry.add_struct = MagicMock()
    return registry

@pytest.fixture
def java_test_file(tmp_path):
    java_code = """
    package com.tostr.test;

    import java.util.List;
    import java.util.ArrayList;

    public class Mathf extends BaseMath implements IMath {
        @Serialized
        public static final double TAU = 6.28;

        public double angleWrap(double radian) {
            return radian % TAU;
        }
    }
    """
    file_path = tmp_path / "Mathf.java"
    file_path.write_bytes(java_code.encode("utf-8"))
    return file_path

def test_java_file_builder_parses_structure(java_test_file, mock_registry):
    builder = JavaFileBuilder(mock_registry)
    
    file_obj = builder.from_path(java_test_file)
    
    assert file_obj.package == "com/tostr/test"
    assert "java/util/List.java#List" in file_obj.imports
    assert "java/util/ArrayList.java#ArrayList" in file_obj.imports
    assert file_obj.body.strip().startswith("package com.tostr.test;")
    
    assert mock_registry.add_struct.call_count == 3
    
    registered_structs = [call.args[0] for call in mock_registry.add_struct.call_args_list]
    
    class_struct = [s for s in registered_structs if s.__class__.__name__ == "BaseClass"][0]
    method_struct = [s for s in registered_structs if s.__class__.__name__ == "BaseMethod"][0]
    field_struct = [s for s in registered_structs if s.__class__.__name__ == "BaseField"][0]
    
    assert class_struct.name == "Mathf"
    assert method_struct.name == "angleWrap"
    assert field_struct.name == "TAU"