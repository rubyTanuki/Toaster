# Creating a New Language Parser in Tostr

This guide is a living document updated after each new language addition. Follow it closely — each section contains hard-won lessons from real implementations. The goal is a complete, correct language addition in one pass.

---

## 0. UID Architecture — Formal Definition (read this before writing any builder)

Every struct has two distinct identities. Conflating them is the single most damaging mistake a language implementation can make (the original Python parser did, and it broke directory hydration, file skeletons, and watcher hash propagation).

### UID (physical identity)

```
directory-uid := <relative-dir-path>                      e.g. src/app/services
file-uid      := <relative-file-path>                     e.g. src/app/services/user_service.py
code-uid      := <file-uid> "#" <code-path>
code-path     := <name> ("." <name>)* [<overload-key>]    e.g. UserService.create(...)
overload-key  := "(" <disambiguator> ")"                  see the overload-key rule below
```

- Top-level class / free function / module field: `<file-uid>#<Name>` → `models.py#User`, `utils.py#clamp(...)`
- Nested members chain with dots **after** the `#`: `models.py#User.validate(...)`, `models.py#Outer.Inner.method(...)`
- The file UID is set by `BaseFileBuilder.from_path` (the relative path) — do not override it.

#### The overload-key rule (read this carefully — it is the whole identity design)

A method/function UID ends in a parenthetical **overload key**. Its job is to carry **exactly what
disambiguates two same-named callables in this language, and nothing else** — never parameter names,
never default values, never annotations, never the return type. Anything that does not change which
call sites resolve to the symbol must NOT be in the UID.

- **Python (no overloading):** the key is the literal `(...)`. A given `(file, class, name)` has exactly
  one definition, so no params are needed to disambiguate. → `models.py#Product.apply_discount(...)`.
- **Java (overloading):** the key is the ordered list of **parameter types**, normalized to a single
  canonical form → `Service.java#Service.process(int, String)`. Renaming a param or changing the return
  type leaves the UID stable; changing a param type or the arity changes it (a genuinely different overload).
  Normalization (`_normalize_java_param_type` in the Java builder): erase generics (`List<String>`→`List`,
  type erasure means you can't overload on them), varargs→array (`String...`→`String[]`), and **strip the
  package qualifier down to the simple/inner-class name** (`java.lang.String`→`String`,
  `com.example.User`→`User`, `java.util.Map.Entry`→`Map.Entry`). Package-stripping (rather than FQN
  *expansion*) is deliberate: a method is declared once so its UID is always self-consistent; the only thing
  canonicalization buys is **stability when an author respells a type**, and stripping makes every spelling
  (`String`/`java.lang.String`, same-package `Order`/`com.example.Order`) agree with **no import map needed**.
  Accepted limitation: two distinct same-simple-name types from different packages collapse together — only
  matters under FQN-in-source overloading on identically-named classes, which is vanishingly rare.
- **Rule of thumb:** if the language has overloading, the key is the canonical param-type list; if not,
  the key is `(...)`.

Why this matters:

1. **Stable identity under cosmetic edits.** Because `id = md5(uid)`, a UID that excludes resolution-
   irrelevant detail means a param rename / default change / annotation change does NOT change the `id`.
   The watcher then updates the body in place and **never has to touch inbound edges**. Encoding param
   names (as the original Python builder did) orphaned the node and dangled every caller on a trivial edit.
2. **Field vs callable is visible in the UID.** `Class.data` is a field; `Class.data(...)` is a method.
   The trailing `(...)` (or `(types)`) makes them distinct UIDs — no exact-match tiebreak needed during
   prefix hydration.
3. **The key is deterministic.** Because Python's key is always `(...)`, logical-name → method translation
   can reconstruct it (`remainder` then `remainder + "(...)"`). Param-name keys could not be reconstructed.

The full signature (param names, types, defaults, return) still lives in the `signature`/`body` **fields**
for `inspect` and LLM context — it just isn't part of identity.

**The prefix invariant (load-bearing):** every struct's UID begins with its parent's UID. Directory → file → class → method, all the way down. Hydration (`uid LIKE ?%` in `get_struct_by_uid`), skeleton generation, and watcher hash propagation (`WHERE uid = <filepath>`) all depend on it. If a builder breaks this invariant, the symptoms appear far away from the cause: directories that don't show their file children, `skeleton` crashing on file paths, hash updates silently no-oping.

### Logical name (language identity)

The dotted name that *source code* uses to refer to a struct: `app.services.user_service.UserService`. Imports, inheritance references, and namespaces are always expressed in this space — source code never mentions file paths. The logical name is **not** the UID.

- Builders MUST set `BaseFile.package` to the file's dotted module/package path. That is the logical root the Registry uses to translate dotted names into physical UIDs (`Registry._resolve_logical_name`).
- `package` is persisted to the DB. Never skip it — watcher-time partial reparses resolve imports against the DB, not memory.
- `registry.get_struct_by_uid()` accepts both forms: exact UIDs match directly; dotted logical names fall back to package translation. Never hand-construct UIDs from dotted names in resolvers — go through the registry.

Both shipped languages follow this definition. Java derives `package` from the `package` declaration (`com.example`); Python derives it from the file's path relative to the project root (`app.services.user_service`). Either way the code-struct UID is `<file-path>#<Code.path><overload-key>`, never the dotted name.

---

## 1. Prerequisites & Environment Setup

Before starting, ensure the Tree-sitter grammar for the language is installed and registered.

### Install Tree-sitter Grammar
1. Find the official tree-sitter package for the language (e.g., `tree-sitter-python`).
2. Add it to `pyproject.toml` under `[project.dependencies]` and to `requirements.txt`.
3. Install it: `pip install tree-sitter-<language>`

### Discovering Tree-sitter Syntax
**Always inspect the actual AST of real files from the grammar version you installed** — documentation and blog posts often describe an older grammar layout that has since changed (see §8 for a concrete example of this biting us). Use this snippet:

```python
from tree_sitter import Parser, Language
import tree_sitter_python as tsp  # replace with your language

LANG = Language(tsp.language())
parser = Parser(LANG)

with open("path/to/sample_file.py", "rb") as f:
    tree = parser.parse(f.read())

def print_tree(node, depth=0):
    prefix = "  " * depth
    print(f"{prefix}[{node.type}] '{node.text.decode()[:40]}' fields={[node.field_name_for_child(i) for i in range(node.child_count)]}")
    for child in node.children:
        print_tree(child, depth + 1)

print_tree(tree.root_node)
```

Run this on multiple real files — `import` statements, class definitions, method calls with receivers, free functions, decorated definitions, relative imports. Print the full node type names and their `field_name` attributes. **Don't guess.**

---

## 2. File Structure

```
src/tostr/languages/<lang>/
├── __init__.py                  # Re-exports builder classes
├── language.py                  # Exports LANGUAGE constant
├── queries.py                   # Tree-sitter SCM queries
├── builders.py                  # Builder implementations
├── dependency_patterns.md       # Documents what IS and IS NOT supported (see §9)
└── default.tostrignore          # Default ignore patterns for this language
```

---

## 3. Registration (do this first — it unblocks testing)

Register in **two places** before writing any builder logic:

**`src/tostr/core/providers.py`** — add to `language_map`:
```python
language_map = {
    "java":   ("tostr.languages.java.builders",   "JavaBuilder",   "tostr.languages.java.resolver.JavaDependencyResolver"),
    "python": ("tostr.languages.python.builders", "PythonBuilder", "tostr.languages.python.resolver.PythonDependencyResolver"),
    "<lang>": ("tostr.languages.<lang>.builders", "<Lang>Builder", "tostr.languages.<lang>.resolver.<Lang>DependencyResolver"),
}
```

Your resolver lives beside your builder in `src/tostr/languages/<lang>/resolver.py` and subclasses
`BaseDependencyResolver` from `tostr.graph.resolver`. (A language with no dependency support can point
straight at `tostr.graph.resolver.BaseDependencyResolver`, as `html` does.)

**`src/tostr/graph/parser.py`** — add to the dependency support set:
```python
self.langs_with_dependency_support = {"java", "python", "<lang>"}
```

If you forget `parser.py`, dependency resolution silently does nothing — no error, no edges, very confusing.

---

## 4. Implementing Builders

Extend from `tostr.core.builders`. The parse order inside `FileBuilder.from_path` is critical (see §8 Phase-order rule).

### `BaseBuilder`
Factory: implements `handles_extension()` and returns the right sub-builder for each struct type.

### `BaseFileBuilder` — most complex, most important

**File identity:** keep the UID assigned by `BaseFileBuilder.from_path` (the relative filepath — never override it with a module name), and set `file_obj.package` to the dotted logical module path (see §0).

**Three-phase parsing order (non-negotiable):**

1. **Phase 1 — Parse imports first.** Build the `alias_map` (`{alias → original_uid}`) and the `imports` list before touching any method bodies. This is required because method dependency names reference aliases, and you need the map ready before children are parsed.

2. **Phase 2 — Parse children** (classes, methods, free functions). These capture raw call-site names into `dependency_names`.

3. **Phase 3 — Normalize aliases** in all descendant `dependency_names` using the `alias_map`. Walk the tree and replace every alias with its original UID in-place.

If you do Phase 2 before Phase 1, aliases in dep names will never be normalized. If you skip Phase 3, imports like `import numpy as np` will leave `np.dot(...)` unresolved.

**Import storage format:**
- `import os` → `["os"]`
- `import os.path` → `["os.path"]`
- `import numpy as np` → `["numpy"]` + `alias_map["np"] = "numpy"`
- `from pathlib import Path` → `["pathlib.Path"]`
- `from pathlib import Path as P` → `["pathlib.Path"]` + `alias_map["P"] = "pathlib.Path"`
- `from math import *` → `["math.*"]`
- `from . import models` (relative) → resolved against current package (see Relative Imports below)

**Relative imports** — resolve at parse time, not resolution time:
```python
# Python example: "from .models import Item" in package "app.services.user"
# module_name field gives ".models" (note leading dot in current tree-sitter-python)
raw = module_node.text.decode()          # ".models"
stripped = raw.lstrip('.')               # "models"
num_dots = len(raw) - len(stripped)      # 1
parts = file_obj.package.split('.')      # ["app", "services", "user"]
base_parts = parts[:-num_dots]           # ["app", "services"]
module_name = ".".join(base_parts + [stripped])  # "app.services.models"
```

**Free functions** — must be added as children of `BaseFile`, not silently dropped. Many languages (Python, Go, JS) have module-scope functions. Failing to capture these means their deps and inbound edges are lost.

**Decorated definitions** — check for wrapper nodes. In Python, a decorated class/function appears as `decorated_definition > class_definition`. Your `_parse_children` must unwrap these:
```python
elif child.type == "decorated_definition":
    for grandchild in child.children:
        if grandchild.type == "class_definition":
            child_instance = class_builder.from_node(grandchild, parent=parent)
            break
        elif grandchild.type == "function_definition":
            child_instance = method_builder.from_node(grandchild, parent=parent)
            break
```

### `BaseClassBuilder`
- Extract `inherits` list (parent class names as strings — resolution happens later).
- Recurse into class body using `FileBuilder._parse_children` (not a separate recursive method — reuse the same child-parsing logic).
- UID format (see §0): `"{file_uid}#{ClassName}"` when the parent is the file, `"{parent_uid}.{ClassName}"` when nested inside another class.

### `BaseMethodBuilder`
- Extract `arity` — **exclude implicit self-like parameters** (`self`, `cls` in Python; nothing to exclude in Java). Callers never pass them, so arity must reflect what callers provide.
  ```python
  first_bare = parameters[0].split('=')[0].split(':')[0].strip() if parameters else ""
  arity_params = parameters[1:] if first_bare in ('self', 'cls') else parameters
  arity = len(arity_params)
  ```
- Run the `DEPENDENCY_QUERY` on the method node (not the file node) using `QueryCursor.matches()`.
- Build `dependency_names` as `[(name, arity, receiver, is_creation)]` tuples. For languages like Python where function calls and class instantiation use identical syntax, set `is_creation=False` for all and let the resolver try type resolution as a fallback.
- UID format (see §0 overload-key rule): `"{file_uid}#{name}{overload_key}"` for free functions, `"{parent_uid}.{name}{overload_key}"` for methods inside a class. The overload key is `(...)` for languages without overloading (Python) and the canonical, normalized param-**type** list for languages with it (Java) — never param names, defaults, annotations, or return type.

### `BaseFieldBuilder`
- **Always parse `field_type`** from the type annotation. If `field_type` is left empty, `self.field.method()` resolution will silently fail — the resolver needs the type to find the class.
- UID format (see §0): `"{file_uid}#{name}"` for module-level fields, `"{parent_uid}.{name}"` for class fields.
- In Python: type annotation is in `assignment.child_by_field_name("type")`.
- In Java: the type precedes the variable name in typed declaration nodes.

---

## 5. Tree-sitter Queries (`queries.py`)

The query must capture `(name, arity, receiver, is_creation)` for every call site.

**Python DEPENDENCY_QUERY (reference implementation):**
```python
DEPENDENCY_QUERY = """
    (call
        function: [
            (identifier) @name
            (attribute
                object: [
                    (identifier) @receiver
                    (attribute) @receiver
                    (call) @receiver
                ]
                attribute: (identifier) @name
            )
        ]
        arguments: (argument_list) @args
    ) @method_call
"""
```

**Critical warning about `(call) @receiver`:** This pattern captures `super()` as a receiver (e.g., `super().method()`). The resolver's Step 1 local search will then find `method` in the current class and create a self-loop. **This is fixed at the resolver level** with the `_use_local_search` hook (§6), NOT by removing the pattern — you need `(call) @receiver` to capture chains like `self.get_client().send()`.

**Java DEPENDENCY_QUERY (reference):**
```python
DEPENDENCY_QUERY = """
(method_invocation
  object: (identifier) @receiver
  name: (identifier) @name
  arguments: (argument_list) @args) @method_call

(object_creation_expression
  type: (type_identifier) @type
  arguments: (argument_list) @args) @object_creation
"""
```

Java separates `is_creation=True` at the query level. Python cannot (same syntax), so Python always sets `is_creation=False` and relies on the resolver's type fallback.

---

## 6. Dependency Resolution (`resolver.py`)

Tostr uses a Strategy pattern. The resolution pipeline runs per method, per captured call:

**Steps (in order):**
1. **`is_creation` shortcut** — if marked as a class instantiation, call `resolve_type(scope, name)` directly and skip to next dep.
2. **Step 1 — Local search** — look for `name` in the method's parent class/file children.
3. **Step 2 — Receiver-based** — resolve the receiver's type, then look up `name` in that type's methods.
4. **Step 3 — Import/inheritance search** — search `_get_potential_lookup_parents()` (namespace wildcard + all imports + inherited classes).
5. **Step 4 — Type resolution fallback** — try `resolve_type(scope, name)` in case it's a class being instantiated without `is_creation=True`.

### The `_use_local_search` Hook (critical for Python)

**Problem:** Step 1 runs before Step 2. When `ReceiverClass.method_name()` is called and the current class ALSO has a method named `method_name`, Step 1 finds the wrong method, sets `resolved=True`, and Step 2 never runs. Result: self-loop instead of cross-file dep.

**Solution:** Add a `_use_local_search(dep_info) -> bool` hook to `BaseDependencyResolver` (returns `True` by default — safe for Java). Override in your resolver to skip local search when a non-self receiver is present:

```python
# In BaseDependencyResolver:
def _use_local_search(self, dep_info: tuple) -> bool:
    return True

# In resolve_method_dependencies Step 1:
if self._use_local_search((name, arity, receiver, is_creation)):
    search_scope = method.parent.children if method.parent else method.children
    for child_set in list(search_scope.values()):
        for child in list(child_set):
            if child.name == name and (not self.strict_arity or getattr(child, "arity", -1) == arity):
                method.add_dependency(child)
                resolved = True
                break
        if resolved: break
    if resolved: continue

# In PythonDependencyResolver:
def _use_local_search(self, dep_info: tuple) -> bool:
    name, arity, receiver, is_creation = dep_info
    if receiver is None:
        return True   # No receiver — could be a local call
    bare = receiver.split('.')[0]
    return bare in ('self', 'cls')  # Only search locally for self/cls calls
```

This also fixes `super().method()` self-loops without any query changes.

### Self-loop Guard (required in `BaseStruct`)

`add_dependency` and `add_fuzzy_dependency` in `models.py` must both begin with:
```python
if target is self:
    return
```
Without this, type annotation resolution (e.g., `parent: BaseStruct` on the `BaseStruct` class itself) will add the class as its own dependency. The guard is universally correct — self-dependencies carry no graph information.

### `PythonDependencyResolver` specifics

```python
class PythonDependencyResolver(BaseDependencyResolver):
    def __init__(self, registry):
        super().__init__(registry)
        self.strict_arity = False  # Python has default/keyword args

    def _use_local_search(self, dep_info):
        name, arity, receiver, is_creation = dep_info
        if receiver is None: return True
        return receiver.split('.')[0] in ('self', 'cls')

    def _resolve_receiver_type(self, method, receiver):
        from tostr.core.models import BaseClass
        if receiver in ('self', 'cls'):
            if isinstance(method.parent, BaseClass):
                return method.parent.uid
            return None
        if receiver.startswith(('self.', 'cls.')):
            field_name = receiver.split('.', 1)[1].split('.')[0]
            if isinstance(method.parent, BaseClass):
                for f in method.parent.fields:
                    if f.name == field_name and f.field_type:
                        return f.field_type
            return None
        return super()._resolve_receiver_type(method, receiver)

    def _get_potential_lookup_parents(self, method):
        parents = super()._get_potential_lookup_parents(method)
        # Named imports like "from utils import format_currency" store
        # "utils.format_currency" in imports. The function's UID is
        # "utils.py#format_currency(amount)" — looking up "utils.format_currency"
        # finds nothing (the params differ, so even logical translation misses).
        # Add the module scope ("utils") so resolve_methods can find the free
        # function as a child of the file struct by name.
        parent_struct = method.parent
        imports = getattr(parent_struct, "imports", [])
        if not imports and parent_struct and parent_struct.parent:
            imports = getattr(parent_struct.parent, "imports", [])
        extra = []
        for imp in imports:
            if not imp.endswith(".*") and "." in imp:
                module_scope = imp.rsplit(".", 1)[0]
                if module_scope not in parents:
                    extra.append(module_scope)
        return parents + extra
```

### Arity

- **Java:** `strict_arity = True`. Java uses method overloading, so arity disambiguates.
- **Python:** `strict_arity = False`. Default args, `*args`, `**kwargs` mean call-site arity rarely matches declaration arity.
- **Rule of thumb:** if the language has overloading, use strict. If not, use loose.

---

## 7. Test Suite

Tests live in `tests/languages/<lang>/`. Cover these cases in order — each one exercises a specific resolution path:

### 7a. Import Parsing Tests (no resolution — just check `file.imports`)
- Simple import: `import os` → `["os"]`
- Dotted import: `import os.path` → `["os.path"]`
- Aliased module import: `import numpy as np` → `["numpy"]` (NOT `"numpy as np"`)
- Named import: `from pathlib import Path` → `["pathlib.Path"]`
- Aliased named import: `from pathlib import Path as P` → `["pathlib.Path"]` (NOT `"pathlib.P"`)
- Wildcard import: `from math import *` → `["math.*"]`
- Relative import: `from .models import Item` in package `pkg.svc` → `["pkg.models.Item"]`

### 7b. Arity Tests
- Method with self/cls: `def bar(self, x, y)` → `arity == 2` (not 3)
- Free function: `def helper(a, b, c)` → `arity == 3`

### 7c. Same-file Resolution
- Free function calls free function (no receiver)
- `self.method()` call within same class
- `cls.method()` classmethod call within same class

### 7d. Cross-file Resolution
- `from module import Class` + `Class()` instantiation → class in `outbound_dependencies`
- `from module import func` + `func()` call → method in `outbound_dependencies`
- `receiver.method()` where receiver type is in imports → method in `outbound_dependencies`

### 7e. Alias Resolution
- Aliased module call: `import utils as u; u.helper()` → `helper` resolved
- Aliased named import call: `from models import User as U; U()` → `User` class resolved

### 7f. Field Type Resolution
- `self.field.method()` where field has a type annotation → method resolved via field type

### 7g. Inheritance
- `self.inherited_method()` where method is defined on parent class
- Class with `inherits` → parent class in `outbound_dependencies`

### 7h. Relative Imports
- `from .sibling import Class; Class()` → resolved correctly
- `from ..parent import Class` (two dots) → resolved correctly

### 7i. Wildcard Imports
- `from module import *; ClassName()` → class in `outbound_dependencies` OR `outbound_dependencies_fuzzy`

### Test fixture (Python example — adapt for your language):
```python
@pytest.fixture
def registry(tmp_path):
    (tmp_path / ".tostr").mkdir()
    # Note: must be [project] section, not top-level key
    (tmp_path / ".tostr" / "config.toml").write_bytes(b'[project]\nlanguage = "python"\n')
    return Registry(project_path=tmp_path, use_cache=False)

def build(registry, tmp_path, filename, code):
    p = tmp_path / filename
    p.write_text(code)
    builder = YourFileBuilder(registry)
    file_obj = builder.from_path(p)
    registry.add_struct(file_obj)
    return file_obj

def resolve_all(registry):
    for file in registry.files:
        file.resolve_dependencies()
```

**Common test failure causes:**
- Config TOML format wrong — `language = "python"` at top level does nothing; it must be under `[project]`.
- Using `source venv/bin/activate && pytest` fails if you renamed `venv` to `.venv` — virtualenvs are not relocatable. Recreate with `python3 -m venv .venv && pip install -e ".[dev]"`.
- Wrong resolver being used — if the TOML config is malformed, the registry defaults to Java resolver and your language-specific resolver is never called. Add a test that asserts `isinstance(registry.get_resolver(), YourResolver)`.

---

## 8. Advanced Lessons & Troubleshooting

### Grammar version drift
Tree-sitter grammars evolve. Field names and node types documented online often differ from the version installed. **Always verify against the installed version** by printing the AST (§1). 

Real example (Python): older grammar docs show a `relative_import` field on `import_from_statement`. The current version does NOT have this — relative imports appear as a `module_name` field containing `.models` (with leading dots). Code that checks for `relative_import_node` will silently miss all relative imports.

### The three-root causes of missing dependency edges

1. **Alias not normalized** — the resolver sees `np.dot()` and finds nothing because `np` isn't a UID. Fix: Phase 1/3 of the three-phase import parsing.

2. **Step 1 bypasses Step 2** — same-named method in current class wins over receiver-based cross-file lookup. Fix: `_use_local_search` hook.

3. **`field_type` is empty** — `self.client.post()` can't resolve because `client`'s type is unknown. Fix: always parse type annotations in `BaseFieldBuilder`.

### Local function-body imports are invisible

```python
def my_method(self):
    from some.module import Helper  # This import is NOT captured
    Helper().do_thing()
```

The file-level import parser only walks top-level AST children (direct children of the module/file node). Imports inside method bodies are invisible. This is an **accepted limitation** — fixing it requires per-method import tracking and significantly more complexity. Document it in `dependency_patterns.md` (§9).

### Free function import resolution gap

`from utils import format_currency` stores `"utils.format_currency"` in `file.imports`. The Step 3 potential parents list includes `"utils.format_currency"` as a parent name. But looking it up returns `None` — the actual UID is `"utils.py#format_currency(...)"`, and the trailing overload key means a bare `utils.py#format_currency` lookup misses. (Because the Python key is the deterministic `(...)`, logical-name translation *can* reconstruct it by trying `remainder` then `remainder + "(...)"`; until that lands, the module-scope workaround below is required.) 

Fix: override `_get_potential_lookup_parents` to also add the module scope (`"utils"`) so `resolve_methods` can find the method as a child of the file struct. See `PythonDependencyResolver._get_potential_lookup_parents` in §6.

### `super()` calls produce self-loops without the hook

`super().method()` is captured with receiver text `"super()"`. Without `_use_local_search`, Step 1 finds `method` in the current class (since it IS defined there, just being delegated upward) and creates a self-dependency. With the hook, Step 1 is skipped, `_resolve_receiver_type` returns `None` for `super()`, and no edge is added — which is correct since super() resolution requires runtime class hierarchy information.

### The `_use_local_search` rule applies to Java too (in principle)

For Java, the same problem can occur: `otherObject.methodName()` where the current class has a method also named `methodName`. Java's `strict_arity = True` mitigates this (arity often differs), but it's still theoretically possible. The base class hook (`_use_local_search` returning `True`) is safe for Java as-is, but if you observe wrong Java deps in this pattern, consider overriding it for Java too.

### `providers.py` vs `registry.py` for resolver registration

The resolver is registered in `providers.py` via `language_map`, not in `registry.py`. `registry.py` calls `LanguageProvider.get_resolver(self)` which reads `language_map`. Don't add switch-case logic to `registry.py`.

### The `tostr init` language flag

`tostr init` defaults to `--language java`. Always pass `--language <lang>` explicitly when testing, or the config will be written as Java and your resolver will never be called.

### Performance: wildcard import scope

Step 3 (`_get_potential_lookup_parents`) adds wildcard entries like `"src.tostr.graph.registry.*"`. `resolve_methods` then calls `get_classes_in_package(package_name)` and iterates all classes. For large packages this is O(classes × methods). Cache `_potential_parents_cache` on the parent struct to avoid recomputing per method call (the base class already does this for `BaseClass` parents).

### Bubble-up storm guard

`add_dependency` already has `if target in self.outbound_dependencies: return` — but without the `if target is self: return` self-guard, recursive bubble-ups can create cycles that exhaust the stack or add O(n²) edges. Always add the self-guard first in both `add_dependency` and `add_fuzzy_dependency`.

---

## 9. Documenting Supported Patterns (`dependency_patterns.md`)

Every language implementation must ship a `dependency_patterns.md` in its language directory. This is a reference for:
- Future agents extending the resolver
- Users who want to understand why a dep edge is missing
- Planning what to add next

Structure it as two sections: **Supported** and **Unsupported (and why)**.

Example unsupported patterns for Python (document the WHY):
- `super().method()` — receiver type requires runtime MRO; tree-sitter gives us `super()` as a call node, not a resolvable type
- Local variable type inference: `x = SomeClass(); x.method()` — would require data-flow analysis across the method body
- Imports inside function bodies — only top-level imports are scanned
- Multi-level chained access beyond one level: `self.a.b.method()` — we resolve `self.a` but not `self.a.b`
- Dynamic dispatch: `getattr(obj, method_name)()` — name is a runtime string

---

## 10. Verification Checklist

- [ ] UIDs follow the §0 formal definition: file UID = relative path, code structs = `file.ext#Code.path<overload-key>` (`(...)` for non-overloading languages, canonical param-type list for overloading ones), prefix invariant holds for every parent/child pair
- [ ] `file.package` set to the dotted logical module path (and verified present in the DB after a parse)
- [ ] Tree-sitter dependency added to `pyproject.toml` and `requirements.txt`
- [ ] `language.py` correctly exports the language object
- [ ] `builders.py` three-phase parsing: imports → children → alias normalization
- [ ] `field_type` correctly parsed in `BaseFieldBuilder`
- [ ] `self`/`cls` (or equivalent) excluded from arity count
- [ ] `queries.py` covers method calls and object creation
- [ ] `resolver.py` has `_use_local_search` hook wired (base class) and overridden (language resolver)
- [ ] `add_dependency` and `add_fuzzy_dependency` have `if target is self: return` guard
- [ ] Language added to `providers.py` `language_map`
- [ ] Language added to `langs_with_dependency_support` in `parser.py`
- [ ] Unit tests cover all 9 categories in §7
- [ ] Test fixture uses correct `[project]` TOML section format
- [ ] `dependency_patterns.md` written for the language
- [ ] Tested on a real open-source project (not just synthetic fixtures) — inspect a core class with MCP and verify cross-file edges are present
