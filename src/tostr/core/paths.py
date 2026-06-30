from __future__ import annotations

from pathlib import Path

class ProjectPaths:
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.config_path = project_path / "tostr.toml"
        self.ignore_path = project_path / ".tostrignore"
        self.lock_path = project_path / "tostr.lock.json"
        self.cache_path = project_path / ".tostr"
        self.db_path = self.cache_path / "cache.db"
        self.log_path = self.cache_path / "tostr.log"
    
    def relative_to_project(self, path: Path) -> Path:
        if not path.is_absolute():
            return path

        try:
            return path.resolve().relative_to(self.project_path.resolve())
        except ValueError:
            import os
            try:
                return path.absolute().relative_to(self.project_path.absolute())
            except ValueError:
                return Path(os.path.relpath(path.resolve(), self.project_path.resolve()))