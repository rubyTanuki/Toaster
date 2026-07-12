from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import json
from tostr.core.models import *

if TYPE_CHECKING:
    from tostr.graph.registry import Registry
    from tree_sitter import Node


def line_bounds(node: "Node") -> tuple[int, int]:
    """A tree-sitter node's span as 1-indexed source lines, clamped to real content.

    Rows are 0-indexed and `end_point` sits *after* the last byte, so a node ending in a
    newline reports column 0 of the next (empty) row — `end_point[0]` then already equals the
    1-indexed last content line (files). A node ending mid-line (code structs) needs `+ 1`.
    """
    end_row, end_col = node.end_point
    end_line = end_row if (end_col == 0 and end_row > node.start_point[0]) else end_row + 1
    return node.start_point[0] + 1, end_line


class BaseBuilder(ABC):
    def __init__(self, registry: Registry):
        self.registry = registry

    def with_type(self, struct_type: str) -> BaseBuilder:
        match(struct_type):
            case "file": return self.build_file()
            case "class": return self.build_class()
            case "method": return self.build_method()
            case "field": return self.build_field()
            case "directory": return self.build_directory()
        
    def build_file(self) -> BaseFileBuilder: return BaseFileBuilder(self.registry)
    
    def build_class(self) -> BaseClassBuilder: return BaseClassBuilder(self.registry)
    
    def build_method(self) -> BaseMethodBuilder: return BaseMethodBuilder(self.registry)
    
    def build_field(self) -> BaseFieldBuilder: return BaseFieldBuilder(self.registry)
    
    def build_directory(self) -> DirectoryBuilder: return DirectoryBuilder(self.registry)
    
class BaseStructBuilder(ABC):
    def __init__(self, registry: Registry):
        self.registry = registry
        
    @abstractmethod
    def from_dict(self, d: dict) -> BaseStruct: pass

class BaseFileBuilder(BaseStructBuilder):
        
    def from_path(self, path: Path, parent: BaseStruct=None) -> BaseFile:
        rel_path = self.registry.paths.relative_to_project(path)
        # logger.debug(f"Building File from path: {rel_path}")
        file_obj = BaseFile(
            name=rel_path.name,
            uid=str(rel_path),
            path=rel_path,
            registry=self.registry
        )
        return file_obj
    
    def from_dict(self, d: dict) -> BaseFile:
        path = self.registry.paths.relative_to_project(Path(d.get("path", ".")))
        # logger.debug(f"Building File from dict with uid: {d.get('uid', d['name'])}")
        return BaseFile(
            uid=d.get("uid", str(d["path"])),
            name=d.get("name", Path(d["path"]).name),
            path=path,
            registry=self.registry,
            description=d.get("description", ""),
            imports=d.get("imports", []),
            body=d.get("body", ""),
            start_line=d.get("start_line", 0),
            end_line=d.get("end_line", 0),
            diff_hash=d.get("diff_hash", ""),
            package=d.get("package", ""),
            _inbound_dependency_strings=json.loads(d.get("inbound_dependency_strings", [])),
            _outbound_dependency_strings=json.loads(d.get("outbound_dependency_strings", [])),
        )

class BaseCodeStructBuilder(BaseStructBuilder):
    def from_node(self, node: Node, parent: BaseStruct=None) -> BaseClass:
        pass

class BaseClassBuilder(BaseCodeStructBuilder):
    def from_dict(self, d: dict) -> BaseClass:
        path = self.registry.paths.relative_to_project(Path(d.get("path", ".")))
        # logger.debug(f"Building Class from dict with uid: {d.get('uid', d['name'])}")
        return BaseClass(
            uid=d.get("uid", d["name"]),
            name=d.get("name", Path(d["path"]).name),
            path=path,
            registry=self.registry,
            description=d.get("description", ""),
            signature=d.get("signature", ""),
            body=d.get("body", ""),
            diff_hash=d.get("diff_hash", ""),
            start_line=d.get("start_line", 0),
            end_line=d.get("end_line", 0),
            inherits=d.get("inherits", []),
            enum_constants=d.get("enum_constants", []),
            _inbound_dependency_strings=json.loads(d.get("inbound_dependency_strings", [])),
            _outbound_dependency_strings=json.loads(d.get("outbound_dependency_strings", [])),
        )

class BaseMethodBuilder(BaseCodeStructBuilder):
    def from_dict(self, d: dict) -> BaseMethod:
        path = self.registry.paths.relative_to_project(Path(d.get("path", ".")))
        # logger.debug(f"Building Method from dict with uid: {d.get('uid', d['name'])}")
        return BaseMethod(
            uid=d.get("uid", d["name"]),
            name=d.get("name", Path(d["path"]).name),
            path=path,
            registry=self.registry,
            description=d.get("description", ""),
            signature=d.get("signature", ""),
            body=d.get("body", ""),
            diff_hash=d.get("diff_hash", ""),
            start_line=d.get("start_line", 0),
            end_line=d.get("end_line", 0),
            arity=d.get("arity", 0),
            _inbound_dependency_strings=json.loads(d.get("inbound_dependency_strings", [])),
            _outbound_dependency_strings=json.loads(d.get("outbound_dependency_strings", [])),
        )

class BaseFieldBuilder(BaseCodeStructBuilder):
    def from_dict(self, d: dict) -> BaseField:
        path = self.registry.paths.relative_to_project(Path(d.get("path", ".")))
        # logger.debug(f"Building Field from dict with uid: {d.get('uid', d['name'])}")
        return BaseField(
            uid=d.get("uid", d["name"]),
            name=d.get("name", Path(d["path"]).name),
            path=path,
            registry=self.registry,
            description=d.get("description", ""),
            signature=d.get("signature", ""),
            body=d.get("body", ""),
            diff_hash=d.get("diff_hash", ""),
            start_line=d.get("start_line", 0),
            end_line=d.get("end_line", 0),
            _inbound_dependency_strings=json.loads(d.get("inbound_dependency_strings", [])),
            _outbound_dependency_strings=json.loads(d.get("outbound_dependency_strings", [])),
        )
    
class DirectoryBuilder(BaseStructBuilder):
    def from_dict(self, d: dict) -> Directory:
        path = self.registry.paths.relative_to_project(Path(d.get("path", ".")))
        # logger.debug(f"Building Directory from dict with path: {path}")
        directory = Directory(path=path, registry=self.registry, uid=d.get("uid"))
        directory.description = d.get("description", "")
        directory.diff_hash = d.get("diff_hash", "")
        
        # Load dependency strings if they exist
        if d.get("inbound_dependency_strings"):
            try:
                directory._inbound_dependency_strings = json.loads(d.get("inbound_dependency_strings"))
            except (json.JSONDecodeError, TypeError):
                pass
        if d.get("outbound_dependency_strings"):
            try:
                directory._outbound_dependency_strings = json.loads(d.get("outbound_dependency_strings"))
            except (json.JSONDecodeError, TypeError):
                pass
                
        return directory

    