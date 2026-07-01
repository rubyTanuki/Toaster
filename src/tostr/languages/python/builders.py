from __future__ import annotations
from tree_sitter import Parser, Node, Query, QueryCursor
from pathlib import Path
import hashlib
import re

from tostr.core.registry import Registry
from tostr.languages.python.language import PYTHON_LANGUAGE
from tostr.core.builders import BaseBuilder, BaseFileBuilder, BaseClassBuilder, BaseMethodBuilder, BaseFieldBuilder, line_bounds
from tostr.languages.python.queries import DEPENDENCY_QUERY
from tostr.core.models import *


def _module_scope_candidates(module_dotted: str) -> list[str]:
    """Normalized file-UID candidates for a dotted module path — the module as a file and
    as a package __init__. Non-existent candidates are pruned at resolution time."""
    base = module_dotted.replace(".", "/")
    return [f"{base}.py", f"{base}/__init__.py"]


def _member_candidates(module_dotted: str, member: str) -> list[str]:
    """UID candidates for `member` imported from `module_dotted`: the class/field form
    (`mod.py#member`) and the callable form (`mod.py#member(...)`)."""
    out = []
    for mod in _module_scope_candidates(module_dotted):
        out.append(f"{mod}#{member}")
        out.append(f"{mod}#{member}(...)")
    return out


class PythonBuilder(BaseBuilder):
    def build_file(self) -> PythonFileBuilder:
        return PythonFileBuilder(self.registry)
    def build_class(self) -> PythonClassBuilder: 
        return PythonClassBuilder(self.registry)
    def build_method(self) -> PythonMethodBuilder: 
        return PythonMethodBuilder(self.registry)
    def build_field(self) -> PythonFieldBuilder: 
        return PythonFieldBuilder(self.registry)

class PythonFileBuilder(BaseFileBuilder):
    def from_path(self, path: Path, parent: BaseStruct=None) -> BaseFile:
        file_obj = super().from_path(path)

        body_bytes = b""
        with open(path, "rb") as f:
            body_bytes = f.read()
        file_obj.body = body_bytes.decode("utf-8")
        file_obj.diff_hash = hashlib.md5(body_bytes).hexdigest()

        parser = Parser(PYTHON_LANGUAGE)
        tree = parser.parse(body_bytes)
        file_obj.node = tree.root_node
        file_obj.start_line, file_obj.end_line = line_bounds(tree.root_node)

        # UID stays the relative filepath (set by BaseFileBuilder.from_path) so all
        # children are prefix-matchable. The dotted module path is the *logical* name,
        # stored on `package` and resolved through Registry's logical-name lookup.
        rel_path = self.registry.relative_to_project(path)
        module_path = ".".join(rel_path.with_suffix("").parts).replace("/", ".").replace("\\", ".")
        file_obj.package = module_path

        # Phase 1: Parse imports first to build alias_map before children are parsed.
        # alias_map: {alias_name -> original_uid} used to normalize call-site names.
        imports = []
        alias_map = {}
        for child in file_obj.node.children:
            if child.type == "import_statement":
                for name_child in child.children_by_field_name("name"):
                    if name_child.type == "aliased_import":
                        original_node = name_child.child_by_field_name("name")
                        alias_node = name_child.child_by_field_name("alias")
                        if original_node:
                            original = original_node.text.decode('utf-8')
                            imports.extend(_module_scope_candidates(original))
                            if alias_node:
                                # Alias binds the module's top-level name; receivers resolve via it.
                                alias_map[alias_node.text.decode('utf-8')] = original.split('.')[0]
                    else:
                        imports.extend(_module_scope_candidates(name_child.text.decode('utf-8')))

            elif child.type == "import_from_statement":
                module_name = ""
                module_node = child.child_by_field_name("module_name")
                relative_import_node = child.child_by_field_name("relative_import")

                if module_node:
                    raw = module_node.text.decode('utf-8')
                    if raw.startswith('.'):
                        # Relative import: ".models" or "..base" — resolve against current package.
                        stripped = raw.lstrip('.')
                        num_dots = len(raw) - len(stripped)
                        if file_obj.package:
                            parts = file_obj.package.split('.')
                            base_parts = parts[:-num_dots] if num_dots <= len(parts) else []
                            module_name = ".".join(base_parts + ([stripped] if stripped else []))
                        else:
                            module_name = stripped
                    else:
                        module_name = raw
                elif relative_import_node:
                    prefix_node = relative_import_node.child_by_field_name("prefix")
                    dotted_name_node = relative_import_node.child_by_field_name("name")
                    dots = prefix_node.text.decode('utf-8') if prefix_node else ""
                    dotted_part = dotted_name_node.text.decode('utf-8') if dotted_name_node else ""
                    num_dots = dots.count('.')
                    if num_dots > 0 and file_obj.package:
                        parts = file_obj.package.split('.')
                        base_parts = parts[:-num_dots] if num_dots <= len(parts) else []
                        module_name = ".".join(base_parts + ([dotted_part] if dotted_part else []))
                    else:
                        module_name = dotted_part

                if module_name:
                    has_wildcard = any(gc.type == "wildcard_import" for gc in child.children)
                    if has_wildcard:
                        imports.extend(f"{mod}.*" for mod in _module_scope_candidates(module_name))
                    else:
                        for gc in child.named_children:
                            if gc.type in {"aliased_import", "dotted_name", "identifier"}:
                                if gc == module_node or gc == relative_import_node:
                                    continue
                                if gc.type == "aliased_import":
                                    name_node = gc.child_by_field_name("name") or gc
                                    alias_node = gc.child_by_field_name("alias")
                                    imp_name = name_node.text.decode('utf-8')
                                    imports.extend(_member_candidates(module_name, imp_name))
                                    if alias_node:
                                        # Alias binds the imported symbol; refs normalize to its real name.
                                        alias_map[alias_node.text.decode('utf-8')] = imp_name
                                else:
                                    imports.extend(_member_candidates(module_name, gc.text.decode('utf-8')))

        file_obj.imports = list(set(imports))

        # Phase 2: Parse children (methods capture raw call-site names/receivers).
        self._parse_children(file_obj.node, file_obj)

        # Phase 3: Normalize aliases in all descendant method dependency_names.
        if alias_map:
            self._normalize_deps(file_obj, alias_map)

        return file_obj

    def _normalize_deps(self, struct: "BaseStruct", alias_map: dict):
        """Post-process all method dependency_names, replacing alias names with original UIDs."""
        for child_set in struct.children.values():
            for child in child_set:
                if isinstance(child, BaseMethod) and child.dependency_names:
                    child.dependency_names = [
                        (
                            alias_map.get(name, name),
                            arity,
                            alias_map.get(receiver, receiver) if receiver else receiver,
                            is_creation,
                        )
                        for name, arity, receiver, is_creation in child.dependency_names
                    ]
                self._normalize_deps(child, alias_map)

    def _parse_children(self, node: Node, parent: BaseStruct):
        class_builder = PythonClassBuilder(self.registry)
        method_builder = PythonMethodBuilder(self.registry)
        field_builder = PythonFieldBuilder(self.registry)

        for child in node.children:
            child_instance = None
            if child.type == "class_definition":
                child_instance = class_builder.from_node(child, parent=parent)
            elif child.type == "function_definition":
                child_instance = method_builder.from_node(child, parent=parent)
            elif child.type == "decorated_definition":
                for grandchild in child.children:
                    if grandchild.type == "class_definition":
                        child_instance = class_builder.from_node(grandchild, parent=parent)
                        break
                    elif grandchild.type == "function_definition":
                        child_instance = method_builder.from_node(grandchild, parent=parent)
                        break
            elif child.type == "expression_statement":
                # Check for assignments
                assignment = None
                if child.named_children and child.named_children[0].type == "assignment":
                    assignment = child.named_children[0]
                if assignment:
                    child_instance = field_builder.from_node(child, parent=parent)
            
            if child_instance:
                parent.add_child(child_instance)
                self.registry.add_struct(child_instance)

class PythonClassBuilder(BaseClassBuilder):
    def from_node(self, node: Node, parent: BaseStruct=None) -> BaseClass:
        body = node.text.decode('utf-8')
        name = ""
        
        # NAME
        name_node = node.child_by_field_name('name')
        if name_node:
            name = name_node.text.decode('utf-8').strip()
            
        # INHERITS
        inherit_strings = []
        superclasses_node = node.child_by_field_name('superclasses')
        if superclasses_node:
            for arg in superclasses_node.named_children:
                inherit_strings.append(arg.text.decode('utf-8').strip())
        
        # SIGNATURE
        signature = f"class {name}"
        if inherit_strings:
            signature += f"({', '.join(inherit_strings)})"
        
        uid = ""
        if isinstance(parent, BaseFile):
            uid = f"{parent.uid}#{name}"
        elif parent:
            uid = f"{parent.uid}.{name}"
        else:
            uid = name

        start_line, end_line = line_bounds(node)
        instance = BaseClass(
            name=name,
            uid=uid,
            registry=self.registry,
            parent=parent,
            path=parent.path if parent else None,
            signature=signature,
            body=body,
            diff_hash=hashlib.md5(node.text).hexdigest(),
            start_line=start_line,
            end_line=end_line,
            node=node,
            inherits=inherit_strings,
        )
        
        # PARSE FOR CHILDREN
        body_node = node.child_by_field_name('body')
        if body_node:
            PythonFileBuilder(self.registry)._parse_children(body_node, instance)
            
        return instance

class PythonMethodBuilder(BaseMethodBuilder):
    def from_node(self, node: Node, parent: BaseStruct=None) -> BaseMethod:
        body = node.text.decode('utf-8')
        name = ""
        
        # NAME
        name_node = node.child_by_field_name('name')
        if name_node:
            name = name_node.text.decode('utf-8').strip()
        
        # PARAMETERS
        # We keep two views: `parameters` carries the full typed/defaulted text for the
        # human-facing signature (surfaced by `inspect`), while `uid_parameters` is the
        # bare-name-only view used to build the UID. Python has no overloading, so names
        # + arity uniquely identify a method; embedding full type hints just bloats the
        # UID (and every skeleton line that renders it).
        parameters = []
        uid_parameters = []
        params_node = node.child_by_field_name('parameters')
        if params_node:
            for param in params_node.named_children:
                # Collapse whitespace/newlines.
                param_text = re.sub(r'\s+', ' ', param.text.decode('utf-8').strip())
                # Full signature param: truncate long type hints / default values.
                parameters.append(param_text if len(param_text) <= 50 else param_text[:47] + "...")
                # UID param: strip the type annotation and default value, keeping any
                # */** prefix (e.g. "x: int = 5" -> "x", "*args" -> "*args").
                uid_parameters.append(param_text.split('=')[0].split(':')[0].strip())

        # Exclude self/cls from arity — callers never pass them.
        first_bare = uid_parameters[0] if uid_parameters else ""
        arity_params = uid_parameters[1:] if first_bare in ('self', 'cls') else uid_parameters
        arity = len(arity_params)

        signature_params = f"({', '.join(parameters)})"
        # Overload key (see §0 of Creating_New_Language_Parser.md): Python has no overloading,
        # so a (file, class, name) is already unique. The UID excludes all params — the literal
        # "(...)" only marks "this is a callable" (distinguishing Class.method(...) from field Class.method)
        # and keeps `id = md5(uid)` stable across param renames / default / annotation edits.
        uid_params = "(...)"

        # SIGNATURE (retains full type detail for `inspect`)
        signature = f"def {name}{signature_params}"

        uid = ""
        if isinstance(parent, BaseFile):
            uid = f"{parent.uid}#{name}{uid_params}"
        elif parent:
            uid = f"{parent.uid}.{name}{uid_params}"
        else:
            uid = f"{name}{uid_params}"

        dependency_names = []
        query = Query(PYTHON_LANGUAGE, DEPENDENCY_QUERY)
        cursor = QueryCursor(query)
        matches = cursor.matches(node)
        for _, captures in matches:
            if "method_call" in captures:
                name_node = captures.get("name")[0]
                dep_name = name_node.text.decode('utf-8').strip()
                args_node = captures.get("args")[0]
                dep_arity = len(args_node.named_children)
                receiver = captures.get("receiver")[0].text.decode('utf-8').strip() if "receiver" in captures else None
                
                # In Python, we don't know if it's object creation or method call.
                # We'll mark is_creation=False and let resolver try both.
                dependency_names.append((dep_name, dep_arity, receiver, False))
        
        start_line, end_line = line_bounds(node)
        return BaseMethod(
            name=name,
            uid=uid,
            registry=self.registry,
            parent=parent,
            path=parent.path if parent else None,
            signature=signature,
            body=body,
            diff_hash=hashlib.md5(node.text).hexdigest(),
            start_line=start_line,
            end_line=end_line,
            node=node,
            arity=arity,
            dependency_names=dependency_names,
        )

class PythonFieldBuilder(BaseFieldBuilder):
    def from_node(self, node: Node, parent: BaseStruct=None) -> BaseField:
        # node is typically an expression_statement containing an assignment
        body = node.text.decode('utf-8')
        name = ""
        field_type = ""
        
        assignment = None
        if node.type == "expression_statement" and node.named_children and node.named_children[0].type == "assignment":
             assignment = node.named_children[0]
        elif node.type == "assignment":
             assignment = node
             
        if assignment:
            left_node = assignment.child_by_field_name('left') or (assignment.named_children[0] if assignment.named_children else None)
            if left_node:
                name = left_node.text.decode('utf-8').strip()
            
            type_node = assignment.child_by_field_name('type')
            if type_node:
                field_type = type_node.text.decode('utf-8').strip()
        
        signature = f"{name}"
        if field_type:
            signature = f"{name}: {field_type}"
            
        uid = ""
        if isinstance(parent, BaseFile):
            uid = f"{parent.uid}#{name}"
        elif parent:
            uid = f"{parent.uid}.{name}"
        else:
            uid = name

        start_line, end_line = line_bounds(node)
        return BaseField(
            name=name,
            uid=uid,
            registry=self.registry,
            parent=parent,
            path=parent.path if parent else None,
            signature=signature,
            body=body,
            diff_hash=hashlib.md5(node.text).hexdigest(),
            start_line=start_line,
            end_line=end_line,
            node=node,
            field_type=field_type,
        )
