from __future__ import annotations
import re
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Optional, List
from loguru import logger

if TYPE_CHECKING:
    from tostr.core.models import BaseStruct, BaseFile, BaseClass, BaseMethod, BaseField
    from tostr.core.registry import Registry


class BaseDependencyResolver:
    """Resolves a struct's call/type references to concrete structs by walking the AST
    graph and the *normalized* import candidates the builders emit. Imports are already
    path-format UIDs (e.g. ``models.py#User`` / ``models.py#User(...)`` / a ``scope.*``
    wildcard), so resolution is a direct ``uid_map`` lookup plus scope-child traversal —
    no raw dotted logical-name translation."""

    def __init__(self, registry: Registry):
        self.registry = registry
        self.strict_arity = True

    # ------------------------------------------------------------------ helpers

    def _enclosing_file(self, struct: BaseStruct) -> Optional[BaseFile]:
        from tostr.core.models import BaseFile
        node = struct
        while node is not None and not isinstance(node, BaseFile):
            node = getattr(node, "parent", None)
        return node

    def _scope_imports(self, scope: BaseStruct) -> List[str]:
        """The import candidates in effect for `scope` (its own, else its parent's)."""
        imports = getattr(scope, "imports", None)
        if imports:
            return imports
        parent = getattr(scope, "parent", None)
        return getattr(parent, "imports", []) if parent else []

    def _simple_name(self, token: str) -> str:
        """Reduce a source type reference to a bare simple name for name-matching:
        strip generics/arrays, then take the final dotted segment. Used only as a
        *fallback* after a direct UID lookup, so it never mangles a real UID."""
        if not token:
            return token
        t = token.split("<")[0].split("[")[0].strip()
        return t.split(".")[-1]

    def _imported_name(self, imp: str) -> Optional[str]:
        """The bare symbol a (non-wildcard) import candidate binds — the member after the
        last '#' (else the final dotted segment), with any '(...)' callable marker stripped.
        Handles both 'pkg/mod.py#User(...)' and 'other.Member'."""
        if not imp or imp.endswith(".*"):
            return None
        tail = imp.rsplit("#", 1)[1] if "#" in imp else imp.rsplit(".", 1)[-1]
        return tail.split("(")[0] or None

    def _descendants(self, struct: BaseStruct) -> List[BaseStruct]:
        out: List[BaseStruct] = []
        stack = list(struct.all_children)
        while stack:
            s = stack.pop()
            out.append(s)
            stack.extend(s.all_children)
        return out

    def _find_named_class(self, scope: BaseStruct, name: str) -> Optional[BaseClass]:
        from tostr.core.models import BaseClass
        for d in self._descendants(scope):
            if isinstance(d, BaseClass) and d.name == name:
                return d
        return None

    # ------------------------------------------------------------- type resolution

    def resolve_type(self, scope: BaseStruct, type_name: str) -> Optional[BaseStruct]:
        """Resolve a type reference (inheritance / field type / object creation) to a
        struct via the enclosing scope and its normalized import candidates."""
        if not type_name:
            return None

        # 1. Direct: already a resolvable normalized UID (or an alias normalized to one).
        dep = self.registry.get_struct_by_uid(type_name)
        if dep:
            return dep

        simple = self._simple_name(type_name)

        # 2. Same enclosing file/scope.
        f = self._enclosing_file(scope)
        if f:
            hit = self._find_named_class(f, simple)
            if hit:
                return hit

        # 3. Imports: a named candidate whose bound name matches, or a wildcard scope to walk.
        for imp in self._scope_imports(scope):
            if imp.endswith(".*"):
                scope_struct = self.registry.resolve_import(imp[:-2])
                if scope_struct:
                    hit = self._find_named_class(scope_struct, simple)
                    if hit:
                        return hit
            elif self._imported_name(imp) == simple:
                dep = self.registry.resolve_import(imp)
                if dep:
                    return dep
        return None

    # ---------------------------------------------------------- method resolution

    def _is_method_match(self, m: BaseStruct, name: str, arity: Optional[int]) -> bool:
        from tostr.core.models import BaseMethod
        return isinstance(m, BaseMethod) and m.name == name and (arity is None or m.arity == arity)

    def resolve_methods(self, name: str, arity: Optional[int], parent_name: Optional[str] = None) -> List[BaseMethod]:
        """Find methods named `name` (optionally by `arity`) within a scope UID. A `scope.*`
        parent walks all descendant methods; a struct UID walks that struct's methods and its
        inheritance chain; no parent falls back to every method in the registry."""
        if parent_name:
            if parent_name.endswith(".*"):
                scope = self.registry.get_struct_by_uid(parent_name[:-2])
                if not scope:
                    return []
                return [m for m in self._descendants(scope) if self._is_method_match(m, name, arity)]

            parent = self.registry.get_struct_by_uid(parent_name)
            if parent:
                return self._resolve_methods_recursive(parent, name, arity, set())
            return []

        return [m for m in self.registry.methods if self._is_method_match(m, name, arity)]

    def _resolve_methods_recursive(self, struct: BaseStruct, name: str, arity: Optional[int], visited: set) -> List[BaseMethod]:
        if struct.uid in visited:
            return []
        visited.add(struct.uid)

        matches = [m for m in struct.methods if m.name == name and (arity is None or m.arity == arity)]
        if matches:
            return matches

        # Inherited methods: resolve each parent type, recurse into it.
        for parent_name in getattr(struct, "inherits", []) or []:
            parent = self.resolve_type(struct, parent_name)
            if parent:
                inherited = self._resolve_methods_recursive(parent, name, arity, visited)
                if inherited:
                    return inherited
        return []

    # ------------------------------------------------ per-method dependency driver

    def _use_local_search(self, dep_info: tuple) -> bool:
        return True

    def resolve_method_dependencies(self, method: BaseMethod):
        """Resolve dependencies for a given method/function."""
        for dep_info in method.dependency_names:
            if len(dep_info) == 2:
                name, arity = dep_info
                receiver, is_creation = None, False
            else:
                name, arity, receiver, is_creation = dep_info

            if is_creation:
                dep = self.resolve_type(method.parent or method, name)
                if dep:
                    method.add_dependency(dep)
                continue

            # --- METHOD RESOLUTION ---
            resolved = False

            # 1. LOCAL SEARCH (same container)
            if self._use_local_search((name, arity, receiver, is_creation)):
                search_scope = method.parent.children if method.parent else method.children
                for child_set in list(search_scope.values()):
                    for child in list(child_set):
                        if child.name == name and (not self.strict_arity or getattr(child, "arity", -1) == arity):
                            method.add_dependency(child)
                            resolved = True
                            break
                    if resolved:
                        break
                if resolved:
                    continue

            # 2. RECEIVER-BASED HEURISTIC
            if receiver:
                receiver_type = self._resolve_receiver_type(method, receiver)
                if receiver_type:
                    dep_type = self.resolve_type(method.parent or method, receiver_type)
                    if dep_type:
                        lookup_arity = arity if self.strict_arity else None
                        candidates = self.resolve_methods(name=name, arity=lookup_arity, parent_name=dep_type.uid)
                        if candidates:
                            method.add_dependency(candidates[0])
                            continue

            # 3. IMPORTED & INHERITED
            potential_parents = self._get_potential_lookup_parents(method)
            all_candidates = []
            lookup_arity = arity if self.strict_arity else None
            for p_name in potential_parents:
                all_candidates.extend(self.resolve_methods(name=name, arity=lookup_arity, parent_name=p_name))

            if len(all_candidates) == 1:
                method.add_dependency(all_candidates[0])
            elif not all_candidates:
                # 4. TYPE RESOLUTION (class instantiation or type reference)
                dep = self.resolve_type(method.parent or method, name)
                if dep:
                    method.add_dependency(dep)
            else:
                # Heuristic: if the receiver matches part of a candidate's class name, prefer it.
                refined_candidates = []
                if receiver:
                    for c in all_candidates:
                        if c.parent and receiver.lower() in c.parent.name.lower():
                            refined_candidates.append(c)

                if len(refined_candidates) == 1:
                    method.add_dependency(refined_candidates[0])
                else:
                    for c in all_candidates:
                        method.add_fuzzy_dependency(c)

    def _resolve_receiver_type(self, method: BaseMethod, receiver: str) -> Optional[str]:
        """Find the type/scope a receiver refers to: a parent field's declared type, or a
        module import whose bound name matches (returns that module's scope UID)."""
        # 1. Parent fields (class fields or module globals).
        if method.parent:
            for field in method.parent.fields:
                if field.name == receiver:
                    return field.field_type

        # 2. A module-scope import bound to this receiver name -> its scope UID.
        for imp in self._scope_imports(method):
            if imp.endswith(".*") or "#" in imp:
                continue  # wildcard scope or member candidate, not a bare module scope
            if PurePosixPath(imp).stem == receiver:
                return imp
        return None

    def _get_potential_lookup_parents(self, method: BaseMethod) -> List[str]:
        """Scope UIDs to search for a called method: the enclosing file (same-module
        free functions), the file's normalized import candidates, and resolved inheritance
        parents. All are real UIDs / `scope.*` markers — no dotted namespaces."""
        parent = method.parent
        if not parent:
            return []

        if getattr(parent, "_potential_parents_cache", None) is not None:
            return parent._potential_parents_cache

        parents: List[str] = []

        # Enclosing file scope (same-module free functions / siblings).
        f = self._enclosing_file(method)
        if f:
            parents.append(f.uid)

        # Normalized import candidates -> resolved to real scope UIDs (bridges the source-root
        # prefix offset so absolute imports in a src-layout still find their target scope).
        for imp in self._scope_imports(parent):
            if imp.endswith(".*"):
                s = self.registry.resolve_import(imp[:-2])
                if s:
                    parents.append(s.uid + ".*")
            else:
                s = self.registry.resolve_import(imp)
                if s:
                    parents.append(s.uid)

        # Inheritance parents, resolved to their UIDs.
        for inh in getattr(parent, "inherits", []) or []:
            t = self.resolve_type(parent, inh)
            if t:
                parents.append(t.uid)

        # Dedupe (order-preserving): anchored + unanchored candidates for the same import
        # resolve to the same scope, and a doubled parent would double its method hits —
        # tripping the single-candidate check into the fuzzy path.
        parents = list(dict.fromkeys(parents))

        if hasattr(parent, "_potential_parents_cache"):
            parent._potential_parents_cache = parents
        return parents


class JavaDependencyResolver(BaseDependencyResolver):
    """Java resolution reuses the base scope-struct logic."""
    pass


class PythonDependencyResolver(BaseDependencyResolver):
    def __init__(self, registry: Registry):
        super().__init__(registry)
        self.strict_arity = False

    def _use_local_search(self, dep_info: tuple) -> bool:
        name, arity, receiver, is_creation = dep_info
        if receiver is None:
            return True
        bare = receiver.split('.')[0]
        return bare in ('self', 'cls')

    def _resolve_receiver_type(self, method: "BaseMethod", receiver: str) -> Optional[str]:
        from tostr.core.models import BaseClass

        # self/cls always refer to the enclosing class.
        if receiver in ('self', 'cls'):
            if isinstance(method.parent, BaseClass):
                return method.parent.uid
            return None

        # self.field or cls.field — look up the first-level field type in the parent class.
        if receiver.startswith(('self.', 'cls.')):
            field_name = receiver.split('.', 1)[1].split('.')[0]
            if isinstance(method.parent, BaseClass):
                for f in method.parent.fields:
                    if f.name == field_name and f.field_type:
                        return f.field_type
            return None

        return super()._resolve_receiver_type(method, receiver)
