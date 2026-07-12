from __future__ import annotations
import pytest
from pathlib import Path
from tostr.config import ProjectConfig # Change this to your actual filename
from tostr.core.models import BaseStruct

@pytest.fixture
def project_root(tmp_path):
    """
    Creates a temporary project structure with a .tostrignore file.
    """
    # Define ignore rules
    ignore_content = [
        "*.log",           # Unanchored extension
        "*.db",            # Unanchored extension
        "/top_level.txt",  # Anchored to root
        "dist/",           # Directory only
        "temp/*.tmp",      # Anchored with wildcard
        "!important.log",  # Negation
    ]
    
    ignore_file = tmp_path / ".tostrignore"
    ignore_file.write_text("\n".join(ignore_content))
    
    # Create the internal .tostr folder to ensure it exists
    (tmp_path / ".tostr").mkdir()
    
    return tmp_path

def test_basic_extension_ignore(project_root):
    config = ProjectConfig(project_root)
    
    # Should ignore .log files anywhere
    assert config.is_ignored(project_root / "debug.log") is True
    assert config.is_ignored(project_root / "src" / "app.log") is True
    
    # Should ignore .db files
    assert config.is_ignored(project_root / "data.db") is True

def test_negation_logic(project_root):
    config = ProjectConfig(project_root)
    
    # *.log is ignored, but !important.log should be kept
    assert config.is_ignored(project_root / "normal.log") is True
    assert config.is_ignored(project_root / "important.log") is False

def test_anchored_vs_unanchored(project_root):
    config = ProjectConfig(project_root)
    
    # /top_level.txt is anchored to root
    assert config.is_ignored(project_root / "top_level.txt") is True
    
    # A file with the same name in a subdirectory should NOT be ignored
    subdir_file = project_root / "src" / "top_level.txt"
    assert config.is_ignored(subdir_file) is False

def test_directory_only_ignore(project_root):
    config = ProjectConfig(project_root)
    
    # Create a directory and a file with the same name
    dist_dir = project_root / "dist"
    dist_dir.mkdir()
    dist_file = dist_dir / "bundle.js"
    
    # Should ignore the directory and its contents
    assert config.is_ignored(dist_dir) is True
    assert config.is_ignored(dist_file) is True
    
    # A file named 'dist' (not a directory) should technically not be ignored 
    # by the 'dist/' rule, but usually, tools treat these safely.
    standalone_file = project_root / "not_a_dir_dist"
    assert config.is_ignored(standalone_file) is False

def test_internal_tostr_ignores(project_root):
    config = ProjectConfig(project_root)
    
    # Hardcoded internal ignores
    assert config.is_ignored(project_root / ".tostr" / "config.toml") is True
    assert config.is_ignored(project_root / ".tostrignore") is True

def test_out_of_bounds_path(project_root, tmp_path_factory):
    config = ProjectConfig(project_root)
    
    # A path completely outside the project root
    external_dir = tmp_path_factory.mktemp("external")
    external_file = external_dir / "external.txt"
    
    # Our logic defaults to True (ignored/skipped) for files outside the root
    assert config.is_ignored(external_file) is True

def test_hardcoded_ignores(project_root):
    config = ProjectConfig(project_root)
    
    # Create directory to test directory-only rules
    pycache = project_root / "__pycache__"
    pycache.mkdir()
    
    git_dir = project_root / ".git"
    git_dir.mkdir()
    
    # Check some hardcoded binaries and system files
    assert config.is_ignored(project_root / ".DS_Store") is True
    assert config.is_ignored(project_root / "my_app.exe") is True
    assert config.is_ignored(project_root / "lib.so") is True
    assert config.is_ignored(pycache) is True
    assert config.is_ignored(git_dir) is True
    
    # Check that it also works for subdirectories
    assert config.is_ignored(project_root / "src" / ".DS_Store") is True
    assert config.is_ignored(project_root / "bin" / "output.bin") is True
