from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC
from typing import Set, List, Dict, Optional, TYPE_CHECKING, ClassVar
import json
import hashlib
from pathlib import Path

from datetime import datetime

from loguru import logger

if TYPE_CHECKING:
    from tostr.graph.registry import Registry
    from tree_sitter import Node


def parse_timestamp(value) -> datetime:
    """Accept either an ISO string (as stored in SQLite) or a datetime, always yielding a datetime."""
    if isinstance(value, datetime):
        return value
    if value:
        try:
            return datetime.fromisoformat(value)
        except (ValueError, TypeError):
            pass
    return datetime.now()


@dataclass
class Note:
    """A note attached and managed by a struct, i.e. a comment outside the source code"""
    content: str = ""
    author: str = ""
    date_added: datetime = field(default_factory=datetime.now)
    date_last_updated: datetime = field(default_factory=datetime.now)
    # Primary key of the backing `notes` row; None until the note has been persisted. This is
    # the handle every note write op (edit/delete) addresses, so an edited note updates its row
    # in place instead of accumulating duplicates.
    id: Optional[int] = None

    def edit(self, new_content: str, author: str):
        self.content = new_content
        self.author = author
        self.date_last_updated = datetime.now()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "author": self.author,
            "date_added": self.date_added.isoformat(),
            "date_last_updated": self.date_last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Note:
        return cls(
            content=d.get("content", "") or "",
            author=d.get("author", "") or "",
            date_added=parse_timestamp(d.get("date_added")),
            date_last_updated=parse_timestamp(d.get("date_last_updated")),
            id=d.get("id"),
        )


@dataclass(eq=False)
class BaseStruct(ABC):
    # IDENTITY
    name: str = ""              # exampleMethod
    uid: str = ""               # namespace.exampleClass#exampleMethod(num1: int) or src/com/example/Example.java
    id: str = field(init=False) # S-1a2b3c4d5e
    description: str = ""
    vector: Optional[List[float]] = None # populated asynchronously by the embedding client
    
    # DEPENDENCIES / GRAPH
    inbound_dependencies: Set[BaseStruct] = field(default_factory=set)
    inbound_dependencies_fuzzy: Set[BaseStruct] = field(default_factory=set) # for fuzzy matching during resolution
    outbound_dependencies: Set[BaseStruct] = field(default_factory=set)
    outbound_dependencies_fuzzy: Set[BaseStruct] = field(default_factory=set) # for fuzzy matching during resolution
    
    _inbound_dependency_strings: List[str] = field(default_factory=list)
    @property
    def inbound_dependency_strings(self):
        if not self._inbound_dependency_strings: self._inbound_dependency_strings = [f"{dep.id}|{dep.uid}" for dep in self.inbound_dependencies] + ['~'+f"{dep.id}|{dep.uid}" for dep in self.inbound_dependencies_fuzzy]
        return self._inbound_dependency_strings
    
    _outbound_dependency_strings: List[str] = field(default_factory=list)
    @property
    def outbound_dependency_strings(self):
        if not self._outbound_dependency_strings: self._outbound_dependency_strings = [f"{dep.id}|{dep.uid}" for dep in self.outbound_dependencies] + ['~'+f"{dep.id}|{dep.uid}" for dep in self.outbound_dependencies_fuzzy]
        return self._outbound_dependency_strings
    
    inbound_dependency_names: Set[str] = field(default_factory=set) # for serialization only, not used for resolution
    outbound_dependency_names: Set[str] = field(default_factory=set) # for serialization only, not used for resolution
    
    # CONTEXT
    registry: Registry = None
    parent: BaseStruct = None
    children: Dict[str, BaseStruct] = field(default_factory=dict)
    path: Path = None
    diff_hash: str = ""

    notes: List[Note] = field(default_factory=list)
    _max_notes: int = 10
    def add_note(self, content: str, author: str):
        """Attach a note in memory. Persisting it is the storage layer's job — go through
        `StructCache.add_note` when the note should survive the process."""
        note = Note(content=content, author=author)
        self.notes.append(note)
        # Cap the ring: oldest notes fall off first so a struct can't grow an unbounded log.
        if len(self.notes) > self._max_notes:
            self.notes = self.notes[-self._max_notes:]
        self.date_last_updated = datetime.now()
        return note

    date_added: datetime = field(default_factory=datetime.now)
    date_last_updated: datetime = field(default_factory=datetime.now)
    
    _IDPREFIX: ClassVar[str] = "S"

    # Caches
    _all_children_cache: Optional[List[BaseStruct]] = field(default=None, init=False, repr=False)
    _type_caches: Dict[type, List[BaseStruct]] = field(default_factory=dict, init=False, repr=False)

    @property
    def extension(self) -> str:
        """File extension this struct originates from (e.g. '.py'), used to route
        language-specific builders/resolvers. Empty when the struct has no path."""
        return Path(self.path).suffix.lower() if self.path else ""

    @property
    def all_children(self):
        if self._all_children_cache is not None: return self._all_children_cache
        all_children = []
        for child_set in self.children.values():
            all_children.extend(child_set)
        self._all_children_cache = all_children
        return all_children
        

    def get_children_by_type(self, type_cls: type) -> List[BaseStruct]:
        if type_cls in self._type_caches:
            return self._type_caches[type_cls]
        
        children = [child for child in self.all_children if isinstance(child, type_cls)]
        self._type_caches[type_cls] = children
        return children
    
    @property
    def directories(self):
        return self.get_children_by_type(Directory)
    
    @property
    def files(self):
        return self.get_children_by_type(BaseFile)
    
    @property
    def methods(self):
        return self.get_children_by_type(BaseMethod)
    
    @property
    def fields(self):
        return self.get_children_by_type(BaseField)
    
    @property
    def classes(self):
        return self.get_children_by_type(BaseClass)
    
    @property
    def edges(self):
        edges = set()
        for dependency in self.outbound_dependencies:
            edges.add((self.id, dependency.id, "depends_on"))
        for dependency in self.outbound_dependencies_fuzzy:
            edges.add((self.id, dependency.id, "depends_on_fuzzy"))
        
        if isinstance(self.parent, BaseStruct):
            edges.add((self.id, self.parent.id, "is_child_of"))
        elif isinstance(self.parent, str):
            edges.add((self.id, self.parent, "is_child_of"))
        
        return edges
    
    @property 
    def impact_score(self) -> int:
        return len(self.inbound_dependencies) + len(self.inbound_dependencies_fuzzy) + len(self.children)
    
    def __post_init__(self):
        id_hash = hashlib.md5(self.uid.encode('utf-8')).hexdigest()[:10]
        self.id = f"{self.__class__._IDPREFIX}-{id_hash}"
        
    def add_child(self, child: BaseStruct):
        type_name = child.__class__.__name__ # e.g., "BaseMethod"
        
        if type_name not in self.children:
            self.children[type_name] = set()
            
        self.children[type_name].add(child)
        child.set_parent(self)
        self._clear_caches()

    def _clear_caches(self):
        self._all_children_cache = None
        self._type_caches = {}
    
    def set_parent(self, parent: BaseStruct):
        if self is parent:
            logger.warning(f"Attempted to set parent of {self} to itself. Skipping to avoid circular reference.")
            return
        self.parent = parent

    def add_dependency(self, target: BaseStruct):
        if target is self:
            return
        if target in self.outbound_dependencies:
            return
        self.outbound_dependencies.add(target)
        self.outbound_dependency_names.add(target.uid)
        target.inbound_dependencies.add(self)
        target.inbound_dependency_names.add(self.uid)
        if isinstance(self.parent, BaseStruct) and isinstance(target.parent, BaseStruct):
            if self.parent != target.parent: 
                self.parent.add_dependency(target.parent)
                
    def add_fuzzy_dependency(self, target: BaseStruct):
        if target is self:
            return
        if target in self.outbound_dependencies_fuzzy:
            return
        self.outbound_dependencies_fuzzy.add(target)
        self.outbound_dependency_names.add('~' + target.uid)
        target.inbound_dependencies_fuzzy.add(self)
        target.inbound_dependency_names.add('~' + self.uid)
        if isinstance(self.parent, BaseStruct) and isinstance(target.parent, BaseStruct):
            if self.parent != target.parent: 
                self.parent.add_fuzzy_dependency(target.parent)
        
    def resolve_dependencies(self):
        # logger.debug(f"Resolving dependencies for {self}")
        for child_set in list(self.children.values()):
            for child in list(child_set):
                child.resolve_dependencies()
        
        if self.registry and self.registry.progress_tracker:
            self.registry.progress_tracker.advance('resolve', 1)

    @classmethod
    def from_dict(cls, d: dict):
        data = d.copy()
        id = data.pop("id", None) 
        instance = cls(**data)
        if id:
            instance.id = id
        return instance
    
    def to_dict(self) -> dict:
        data = {
            "id": self.id,
            "name": self.name,
            "uid": self.uid,
            "type": "struct",
            "path": str(self.path) if self.path else ".",
            "description": self.description,
            "diff_hash": self.diff_hash,
            "inbound_dependency_strings": self.inbound_dependency_strings,
            "outbound_dependency_strings": self.outbound_dependency_strings,
            # `notes` is routed to the `notes` table by the storage layer (like `vector`),
            # not to a `structs` column — see StructCache.save_to_cache.
            "notes": [note.to_dict() for note in self.notes],
            "date_added": self.date_added.isoformat(),
            "date_last_updated": self.date_last_updated.isoformat()
        }
        if self.vector is not None:
            data["vector"] = self.vector
        return data
    
    def to_json(self, indent=0):
        return json.dumps(self.to_dict(), indent=indent)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if not isinstance(other, BaseStruct):
            return False
        return self.id == other.id
    
    def __str__(self):
        return f"<{self.__class__.__name__}: {self.uid}>"
    __repr__=__str__
    
@dataclass(eq=False)
class Directory(BaseStruct):
    _IDPREFIX: ClassVar[str] = "D"
    diff_hash: str = ""
    
    def __init__(self, path, registry=None, parent=None, uid=None):
        uid = uid or str(path)
        name = path.name
        # Handle the root directory case where path.name is empty for Path('.')
        if not name and registry and registry.project_path and str(path) == ".":
            name = registry.project_path.name

        super().__init__(name=name, path=path, uid=uid, registry=registry, parent=parent)
    
    def to_dict(self) -> dict:
        data = super().to_dict()
        data["type"] = "directory"
        data["diff_hash"] = self.diff_hash
        return data

@dataclass(eq=False)
class BaseFile(BaseStruct):
    _IDPREFIX: ClassVar[str] = "F"
    
    imports: List[str] = field(default_factory=list)
    package: str = ""
    body: str = ""
    # Line bounds (1-indexed source lines, matching BaseCodeStruct). Duplicated here
    # rather than inherited so a file can report its span without being a BaseCodeStruct —
    # the same deliberate redundancy already used for `body`/`node`. Directories, which have
    # no source extent, intentionally omit these.
    start_line: int = 0
    end_line: int = 0
    diff_hash: str = ""
    node: "Node" = None

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["type"] = "file"
        data["imports"] = self.imports
        data["body"] = self.body
        data["start_line"] = self.start_line
        data["end_line"] = self.end_line
        data["diff_hash"] = self.diff_hash
        data["package"] = self.package
        return data

@dataclass(eq=False) 
class BaseCodeStruct(BaseStruct):
    
    signature: str = ""         # public static int add(int num1, int num2) or class <T> Example extends BaseClass
    body: str = ""              # signature + method body or class body for hashing and LLM context
    start_line: int = 0         # 1-indexed source line (matches editors/tracebacks), not the 0-indexed tree-sitter row
    end_line: int = 0
    node: Node = None         # Optional reference to the tree-sitter node for advanced processing (e.g., skeletonization)
    
    def to_dict(self) -> dict:
        data = super().to_dict()
        data["type"] = "codestruct"
        data["signature"] = self.signature
        data["body"] = self.body
        data["start_line"] = self.start_line
        data["end_line"] = self.end_line
        return data
    
@dataclass(eq=False)
class BaseClass(BaseCodeStruct):
    _IDPREFIX: ClassVar[str] = "C"
    enum_constants: Optional[List[str]] = None
    inherits: List[str] = field(default_factory=list) # list of parent class UIDs for inheritance relationships
    
    _potential_parents_cache: Optional[List[str]] = field(default=None, init=False, repr=False)

    def _clear_caches(self):
        super()._clear_caches()
        self._potential_parents_cache = None

    @property
    def needs_description(self) -> bool:
        if not self.description:
            return True
        for child_set in self.children.values():
            for child in child_set:
                if not child.description and not isinstance(child, BaseField):
                    return True
        return False

    @property
    def imports(self) -> List[str]:
        return self.parent.imports
    
    def resolve_dependencies(self):
        # logger.debug(f"Resolving dependencies for {self}")
        resolver = self.registry.get_resolver(self.extension)

        # resolve child dependencies
        super().resolve_dependencies()
        
        # resolve import dependencies
        for imp in self.imports:
            if not imp.endswith(".*"):
                import_dependency = self.registry.resolve_import(imp)
                if import_dependency:
                    self.add_dependency(import_dependency)
        
        # resolve inheritance dependencies
        if self.inherits:
            for parent_class in self.inherits:
                dep = resolver.resolve_type(self, parent_class)
                if dep:
                    self.add_dependency(dep)
        
        # resolve field types
        for field in self.fields:
            dep = resolver.resolve_type(self, field.field_type)
            if dep:
                self.add_dependency(dep)
        
        # logger.debug(f"Resolved dependencies for {self.name}: {self.outbound_dependency_strings}")

    def skeletonize(self) -> str:
        if not hasattr(self, 'node') or not self.node:
            raise ValueError("Node reference is required for skeletonization.")
        
        result_bytes = self.node.text
        start_byte = self.node.start_byte
        
        children_to_replace = []
        for child_set in self.children.values():
            if child_set and isinstance(next(iter(child_set)), BaseCodeStruct):
                children_to_replace.extend(child_set)
        children_to_replace.sort(key=lambda x: x.node.start_byte, reverse=True)
        
        for child in children_to_replace:
            if not child.description:
                continue
            
            rel_start = child.node.start_byte - start_byte
            rel_end = child.node.end_byte - start_byte
            
            method_skeleton = f"{child.id} | {child.signature}\n// {child.description}"
            
            skeleton_bytes = method_skeleton.encode('utf-8')
            result_bytes = result_bytes[:rel_start] + skeleton_bytes + result_bytes[rel_end:]
            
        return result_bytes.decode('utf-8')
    
    def to_dict(self) -> dict:
        data = super().to_dict()
        data["type"] = "class"
        if self.enum_constants:
            data["enum_constants"] = self.enum_constants
        data["inherits"] = self.inherits
        return data
    
@dataclass(eq=False)
class BaseMethod(BaseCodeStruct):
    _IDPREFIX: ClassVar[str] = "M"
    
    arity: int = 0
    # List of (name, arity, receiver, is_creation)
    dependency_names: Optional[List[tuple]] = field(default_factory=list)
    
    children: dict = field(init=False, repr=False, default_factory=dict)
    
    # @abstractmethod
    # def _parse_dependencies(self):
    #     pass

    def resolve_dependencies(self):
        logger.debug(f"Resolving dependencies for method {self.uid}")
        resolver = self.registry.get_resolver(self.extension)
        resolver.resolve_method_dependencies(self)
        
        if self.registry and self.registry.progress_tracker:
            self.registry.progress_tracker.advance('resolve', 1)

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["type"] = "method"
        data["arity"] = self.arity
        data["dependency_names"] = self.dependency_names
        return data

@dataclass(eq=False)
class BaseField(BaseCodeStruct):
    _IDPREFIX: ClassVar[str] = "V"
    
    field_type: str = ""
    children: dict = field(init=False, repr=False, default_factory=dict)

    # @abstractmethod
    # def _parse_dependencies(self):
    #     pass

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["type"] = "field"
        data["field_type"] = self.field_type
        return data
