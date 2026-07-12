from __future__ import annotations
from tostr.core.models import *

from enum import IntEnum
from loguru import logger
from dataclasses import dataclass, field
from typing import List, Optional, Union

@dataclass
class InspectResult:
    id: str
    uid: str
    filepath: str
    signature: str = ""
    description: str = ""
    inbound_edges: List[str] = field(default_factory=list)
    outbound_edges: List[str] = field(default_factory=list)
    fields: List[InspectResult] = field(default_factory=list)
    methods: List[InspectResult] = field(default_factory=list)
    classes: List[InspectResult] = field(default_factory=list)
    files: List[InspectResult] = field(default_factory=list)
    directories: List[InspectResult] = field(default_factory=list)
    body: Optional[str] = None
    start_line: int = 0
    end_line: int = 0
    type: str = "struct"

@dataclass
class SkeletonResult:
    id: str
    uid: str
    type: str
    children: List[SkeletonResult] = field(default_factory=list)

@dataclass
class SearchResult:
    id: str
    uid: str
    type: str
    distance: float

class tost:
    
    @classmethod
    def dump_skeleton(
        cls, 
        obj: "BaseStruct",
        files_only: bool = True,
        depth: int = 7,
    ) -> SkeletonResult:
        children = []
        if depth > 0:
            if obj.files:
                for f in obj.files:
                    children.append(cls.dump_skeleton(f, files_only=files_only, depth=depth-1))
            if obj.directories:
                for d in obj.directories:
                    if d is obj:
                        logger.warning(f"Skipping dumping directory {d} as it is the same as its parent {obj}, likely to avoid circular reference.")
                        continue
                    children.append(cls.dump_skeleton(d, files_only=files_only, depth=depth-1))
            if obj.classes and not files_only:
                for c in obj.classes:
                    children.append(cls.dump_skeleton(c, files_only=files_only, depth=depth-1))
            # Surface top-level (module-level) functions that live directly on a file,
            # e.g. Python modules of free functions with no enclosing class. Class members
            # are deliberately excluded: the class remains the floor for its own subtree.
            if isinstance(obj, BaseFile) and not files_only:
                for m in obj.methods:
                    children.append(SkeletonResult(id=m.id, uid=m.uid, type=type(m).__name__))

        return SkeletonResult(
            id=obj.id,
            uid=obj.uid,
            type=obj.__class__.__name__,
            children=children
        )
    
    @classmethod
    def dump(
        cls, 
        obj: "BaseStruct", 
        depth: int = 2, # recursion depth for children
        include_body: bool = False, 
    ) -> InspectResult:
        result = InspectResult(
            id=obj.id,
            uid=obj.uid,
            filepath=str(obj.path) if obj.path else "",
            signature=getattr(obj, 'signature', ""),
            description=obj.description,
            inbound_edges=obj.inbound_dependency_strings,
            outbound_edges=obj.outbound_dependency_strings,
            # Code structs and files both carry a body; emit it on request. Directories
            # have none, so the truthiness check keeps them out without special-casing.
            body=getattr(obj, "body", None) if include_body else None,
            start_line=getattr(obj, 'start_line', 0),
            end_line=getattr(obj, 'end_line', 0),
            type=obj.__class__.__name__
        )

        if depth > 0:
            if obj.files:
                result.files = [cls.dump(f, depth=depth-1, include_body=False) for f in obj.files]
            if obj.directories:
                result.directories = [cls.dump(d, depth=depth-1, include_body=False) for d in obj.directories]
            if obj.classes:
                result.classes = [cls.dump(c, depth=depth-1, include_body=False) for c in obj.classes]
            if obj.fields:
                result.fields = [cls.dump(f, depth=0, include_body=False) for f in obj.fields]
            if obj.methods:
                result.methods = [cls.dump(m, depth=0, include_body=False) for m in obj.methods]
            
        return result
