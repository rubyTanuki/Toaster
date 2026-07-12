from __future__ import annotations
from tree_sitter import Parser, Node, Query, QueryCursor
from pathlib import Path
import hashlib
import re

from tostr.graph.registry import Registry
from tostr.languages.java.language import JAVA_LANGUAGE
from tostr.core.builders import BaseBuilder, BaseFileBuilder, BaseClassBuilder, BaseMethodBuilder, BaseFieldBuilder, line_bounds
from tostr.languages.java.queries import DEPENDENCY_QUERY
from tostr.core.models import *

def _strip_java_package(t: str) -> str:
    """Drop package qualifiers from a (non-generic, non-array) type, keeping the simple class name
    plus any enclosing class names. This canonicalizes the *spelling* of a type so the UID is stable
    when an author respells it — `String` / `java.lang.String`, `User` / `com.example.User`, and a
    same-package `Order` / `com.example.Order` all collapse to one form, with no import map needed.

    Heuristic: by Java convention package segments are lowercase-initial and type names are
    uppercase-initial, so we keep everything from the first uppercase-initial segment onward. This
    preserves inner classes (`java.util.Map.Entry` -> `Map.Entry`) while stripping the package.

    Known limitation (documented in dependency_patterns.md): two distinct types that share a simple
    name from different packages (`java.util.List` vs `java.awt.List`) collapse to the same token, so
    overloads distinguished only by that would merge. This requires FQN-in-source overloading on
    identically-named classes — vanishingly rare, and the false merge is preferred over the identity
    instability that partial FQN expansion would cause."""
    if '.' not in t:
        return t
    parts = t.split('.')
    for i, p in enumerate(parts):
        if p[:1].isupper():
            return '.'.join(parts[i:])
    return parts[-1]  # all-lowercase (e.g. an unconventional type) — fall back to the last segment

def _normalize_java_param_type(raw: str) -> str:
    """Canonicalize a declared parameter type into its overload-dispatch form for the UID
    (see §0 overload-key rule). Java overload resolution uses *erased* types, so:
      - generic arguments are erased     (List<String>      -> List)        — can't overload on them
      - varargs collapse to an array     (String...         -> String[])    — varargs IS an array for dispatch
      - package qualifiers are stripped  (java.lang.String  -> String)      — stable spelling (see _strip_java_package)
      - whitespace is collapsed
    Unlike the display signature, the UID type is NOT truncated (truncation would risk collisions)."""
    t = re.sub(r'\s+', ' ', raw).strip()
    if t.endswith('...'):
        t = t[:-3].strip() + '[]'
    # Erase generics by repeatedly stripping the innermost <...> until none remain (handles nesting).
    while '<' in t:
        new_t = re.sub(r'<[^<>]*>', '', t)
        if new_t == t:
            break  # unbalanced angle brackets — stop rather than loop forever
        t = new_t
    # Peel off (possibly multi-dimensional) array brackets, strip the package off the base, reattach.
    arr = ''
    while t.endswith('[]'):
        arr += '[]'
        t = t[:-2].strip()
    return _strip_java_package(t) + arr

def _java_import_candidates(fqn: str, is_wildcard: bool) -> list[str]:
    """Normalize a Java import FQN to path-format UID candidates. A wildcard package import
    `com.example.*` -> the package-directory scope marker `com/example.*`; a type import
    `com.example.User` -> the file member UID `com/example/User.java#User`."""
    parts = fqn.split(".")
    if is_wildcard:
        return [f"{'/'.join(parts)}.*"]
    class_name = parts[-1]
    pkg_path = "/".join(parts[:-1])
    file_uid = f"{pkg_path}/{class_name}.java" if pkg_path else f"{class_name}.java"
    return [f"{file_uid}#{class_name}"]


class JavaBuilder(BaseBuilder):
    def build_file(self) -> JavaFileBuilder:
        return JavaFileBuilder(self.registry)
    def build_class(self) -> JavaClassBuilder: 
        return JavaClassBuilder(self.registry)
    def build_method(self) -> JavaMethodBuilder: 
        return JavaMethodBuilder(self.registry)
    def build_field(self) -> JavaFieldBuilder: 
        return JavaFieldBuilder(self.registry)
    
    
class JavaFileBuilder(BaseFileBuilder):
    def from_path(self, path: Path, parent: BaseStruct=None) -> BaseFile:
        file_obj = super().from_path(path)
    
        imports = []
        body_bytes = b""
        
        with open(path, "rb") as f:
            body_bytes = f.read()
        file_obj.body = body_bytes.decode("utf-8")
        file_obj.diff_hash = hashlib.md5(body_bytes).hexdigest() # hash of the file's text body
        
        parser = Parser(JAVA_LANGUAGE)
        tree = parser.parse(body_bytes)
        file_obj.node = tree.root_node
        file_obj.start_line, file_obj.end_line = line_bounds(tree.root_node)

        # parse tree-sitter tree for children and imports
        class_builder = JavaClassBuilder(self.registry)
        method_builder = JavaMethodBuilder(self.registry)
        field_builder = JavaFieldBuilder(self.registry)
        enum_builder = JavaEnumBuilder(self.registry)
        for child in file_obj.node.children:
            if child.type == "package_declaration":
                for grandchild in child.children:
                    if grandchild.type in {"scoped_identifier", "identifier"}:
                        # Store the package in normalized path form (com.example -> com/example);
                        # it is display metadata only — resolution keys off the file's own path.
                        file_obj.package = grandchild.text.decode('utf-8').replace(".", "/")
                        break

            if child.type == "import_declaration":
                is_wildcard = any(gc.type == "asterisk" for gc in child.children)
                for grandchild in child.children:
                    if grandchild.type in {"scoped_identifier", "identifier"}:
                        fqn = grandchild.text.decode('utf-8')
                        imports.extend(_java_import_candidates(fqn, is_wildcard))
                        
            child_instance = None
            if child.type == "class_declaration" or child.type == "interface_declaration":
                child_instance = class_builder.from_node(child, parent=file_obj)
            elif child.type == "enum_declaration":
                child_instance = enum_builder.from_node(child, parent=file_obj)
            elif child.type == "method_declaration":
                child_instance = method_builder.from_node(child, parent=file_obj)
            elif child.type == "field_declaration":
                child_instance = field_builder.from_node(child, parent=file_obj)
            if child_instance:
                file_obj.add_child(child_instance)
                self.registry.add_struct(child_instance)
        
        file_obj.imports = imports
        
        return file_obj
        
        
class JavaClassBuilder(BaseClassBuilder):
    def from_node(self, node: Node, parent: BaseStruct=None) -> BaseClass:
        body = node.text.decode('utf-8')
        sig_parts = []
        name = ""
        
        # MODIFIERS
        for child in node.children:
            if child.type == 'modifiers':
                for mod_child in child.children:
                    if 'comment' not in mod_child.type:
                        sig_parts.append(mod_child.text.decode('utf-8').strip())
                break
        
        sig_parts.append("class")
        
        # NAME
        name_node = node.child_by_field_name('name')
        if name_node:
            name = name_node.text.decode('utf-8').strip()
            
        # TYPE PARAMETERS
        type_params_node = node.child_by_field_name('type_parameters')
        type_params_string = ""
        if type_params_node:
            type_params_string = type_params_node.text.decode('utf-8').strip()
        
        # INHERITS
        inherit_strings = []
        inherits_node = node.child_by_field_name('superclass')
        if inherits_node:
            identifier_node = inherits_node.named_children[0]
            inherit_strings.append(identifier_node.text.decode('utf-8').strip())
        interfaces_node = node.child_by_field_name('interfaces')
        if interfaces_node:
            type_list_node = interfaces_node.named_children[0]
            for type_node in type_list_node.named_children:
                inherit_strings.append(type_node.text.decode('utf-8').strip())
        
        # SIGNATURE
        sig_prefix = " ".join(sig_parts).replace('\n', ' ')
        signature = f"{sig_prefix} {name}{type_params_string}"
        
        uid = ""
        if isinstance(parent, BaseFile):
            uid = f"{parent.uid}#{name}"
        else:
            uid = f"{parent.uid}.{name}"

        start_line, end_line = line_bounds(node)
        instance = BaseClass(
           # BaseStruct
            name=name,
            uid=uid,
            registry=self.registry,
            parent=parent,
            path=parent.path if parent else None,
            
            # BaseCodeStruct
            signature=signature,
            body=body,
            diff_hash=hashlib.md5(node.text).hexdigest(),
            start_line=start_line,
            end_line=end_line,
            node=node,
            
            # BaseClass
            inherits=inherit_strings,
        )
        
        # PARSE FOR CHILDREN
        class_builder = JavaClassBuilder(self.registry)
        method_builder = JavaMethodBuilder(self.registry)
        field_builder = JavaFieldBuilder(self.registry)
        enum_builder = JavaEnumBuilder(self.registry)
        
        body_node = node.child_by_field_name('body')
        if body_node:
            for child in body_node.children:
                child_instance = None
                if child.type == "class_declaration" or child.type == "interface_declaration":
                    child_instance = class_builder.from_node(child, parent=instance)
                elif child.type == "enum_declaration":
                    child_instance = enum_builder.from_node(child, parent=instance)
                elif child.type == "constructor_declaration":
                    child_instance = method_builder.from_node(child, parent=instance)
                elif child.type == "method_declaration":
                    child_instance = method_builder.from_node(child, parent=instance)
                elif child.type == "field_declaration":
                    child_instance = field_builder.from_node(child, parent=instance)
                if child_instance:
                    instance.add_child(child_instance)
                    self.registry.add_struct(child_instance)
            
        return instance
        
        

class JavaMethodBuilder(BaseMethodBuilder):
    def from_node(self, node: Node, parent: BaseStruct=None) -> BaseMethod:
        body = node.text.decode('utf-8')
        sig_parts = []
        name = ""
        
        # MODIFIERS
        for child in node.children:
            if child.type == 'modifiers':
                for mod_child in child.children:
                    if 'comment' not in mod_child.type:
                        sig_parts.append(mod_child.text.decode('utf-8').strip())
        
        # TYPE PARAMETERS
        type_params_node = node.child_by_field_name('type_parameters')
        if type_params_node:
            sig_parts.append(type_params_node.text.decode('utf-8').strip())
        
        # TYPE
        type_node = node.child_by_field_name('type')
        if type_node:
            sig_parts.append(type_node.text.decode('utf-8').strip())
        
        # NAME
        name_node = node.child_by_field_name('name')
        if name_node:
            name = name_node.text.decode('utf-8').strip()
        
        # PARAMETERS
        # `parameters` holds the display form (truncated) for the signature; `uid_param_types`
        # holds the normalized, untruncated overload-dispatch types for the UID (see §0).
        parameters = []
        uid_param_types = []
        params_node = node.child_by_field_name('parameters')
        if params_node:
            for param_child in params_node.named_children:
                if param_child.type == 'formal_parameter':
                    param_type_node = param_child.child_by_field_name('type')
                    if param_type_node:
                        raw_type = param_type_node.text.decode('utf-8').strip()
                        uid_param_types.append(_normalize_java_param_type(raw_type))
                        # Display form: collapse whitespace/newlines and truncate long types
                        param_text = re.sub(r'\s+', ' ', raw_type)
                        if len(param_text) > 50:
                            param_text = param_text[:47] + "..."
                        parameters.append(param_text)

        arity = len(parameters)
        parameters_string = f"({', '.join(parameters)})"
        # Overload key: ordered, normalized param TYPES (never names/return) so overloads get
        # distinct UIDs while param renames / return-type changes keep `id` stable.
        uid_params = f"({', '.join(uid_param_types)})"

        # SIGNATURE
        sig_prefix = " ".join(sig_parts).replace('\n', ' ')
        signature = f"{sig_prefix} {name}{parameters_string}".strip()

        uid = ""
        if isinstance(parent, BaseFile):
            uid = f"{parent.uid}#{name}{uid_params}"
        else:
            uid = f"{parent.uid}.{name}{uid_params}"
        
        dependency_names = []
        
        query = Query(JAVA_LANGUAGE, DEPENDENCY_QUERY)
        cursor = QueryCursor(query)
        matches = cursor.matches(node)
        for _, captures in matches:
            if "method_call" in captures:
                name_node = captures.get("name")[0]
                dep_name = name_node.text.decode('utf-8').strip()
                args_node = captures.get("args")[0]
                dep_arity = len(args_node.named_children)
                receiver = captures.get("receiver")[0].text.decode('utf-8').strip() if "receiver" in captures else None
                dependency_names.append((dep_name, dep_arity, receiver, False))
            elif "object_creation" in captures:
                type_node = captures.get("type")[0]
                dep_name = type_node.text.decode('utf-8').strip()
                args_node = captures.get("args")[0]
                dep_arity = len(args_node.named_children)
                dependency_names.append((dep_name, dep_arity, None, True))
        
        start_line, end_line = line_bounds(node)
        return BaseMethod(
            # BaseStruct
            name=name,
            uid=uid,
            registry=self.registry,
            parent=parent,
            path=parent.path if parent else None,
            
            # BaseCodeStruct
            signature=signature,
            body=body,
            diff_hash=hashlib.md5(node.text).hexdigest(),
            start_line=start_line,
            end_line=end_line,
            node=node,
            
            # BaseMethod
            arity=arity,
            dependency_names=dependency_names,
        )

class JavaFieldBuilder(BaseFieldBuilder):
    def from_node(self, node: Node, parent: BaseStruct=None) -> BaseField:
        body = node.text.decode('utf-8')
        sig_parts = []
        name = ""
        
        # MODIFIERS
        for child in node.children:
            if child.type == 'modifiers':
                for mod_child in child.children:
                    if 'comment' not in mod_child.type:
                        sig_parts.append(mod_child.text.decode('utf-8').strip())
        
        # TYPE
        type_node = node.child_by_field_name('type')
        field_type = type_node.text.decode('utf-8').strip() if type_node else ""
        if field_type:
            sig_parts.append(field_type)
        
        # NAME
        declarator_node = node.child_by_field_name('declarator')
        if declarator_node:
            name_node = declarator_node.child_by_field_name('name')
            if name_node:
                name = name_node.text.decode('utf-8').strip()
                sig_parts.append(name)
        
        signature = " ".join(sig_parts).replace('\n', ' ')
        
        uid = ""
        if isinstance(parent, BaseFile):
            uid = f"{parent.uid}#{name}"
        else:
            uid = f"{parent.uid}.{name}"

        start_line, end_line = line_bounds(node)
        return BaseField(
            # BaseStruct
            name=name,
            uid=uid,
            registry=self.registry,
            parent=parent,
            path=parent.path if parent else None,
            
            # BaseCodeStruct
            signature=signature,
            body=body,
            diff_hash=hashlib.md5(node.text).hexdigest(),
            start_line=start_line,
            end_line=end_line,
            node=node,
            
            # BaseField
            field_type=field_type,
        )

class JavaEnumBuilder(JavaClassBuilder):
    def from_node(self, node: Node, parent: BaseStruct=None) -> BaseClass:
        # super().from_node(node, parent)
        return None
        # TODO: extract enum values and add them to enum_constants in a class