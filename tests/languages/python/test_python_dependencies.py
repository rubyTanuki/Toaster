from __future__ import annotations
import pytest
from pathlib import Path
from tostr.graph.registry import Registry
from tostr.core.paths import ProjectPaths
from tostr.languages.python.builders import PythonFileBuilder
from tostr.core.models import BaseStruct


@pytest.fixture
def registry(tmp_path):
    (tmp_path / ".tostr").mkdir()
    (tmp_path / "tostr.toml").write_bytes(b'[project]\nlanguage = "python"\n')
    return Registry(ProjectPaths(tmp_path))


def build(registry, tmp_path, filename, code):
    p = tmp_path / filename
    p.write_text(code)
    builder = PythonFileBuilder(registry)
    file_obj = builder.from_path(p)
    registry.add_struct(file_obj)
    for struct in list(registry.uid_map.values()):
        if struct not in [file_obj]:
            pass  # already added during _parse_children via registry.add_struct
    return file_obj


def resolve_all(registry):
    for file in registry.files:
        file.resolve_dependencies()


# ---------------------------------------------------------------------------
# Import parsing correctness
# ---------------------------------------------------------------------------

def test_simple_import(tmp_path, registry):
    code = "import os\nimport os.path\n"
    f = build(registry, tmp_path, "a.py", code)
    # Module imports normalize to file-scope UID candidates.
    assert "os.py" in f.imports
    assert "os/path.py" in f.imports


def test_aliased_module_import_stores_original(tmp_path, registry):
    code = "import collections as col\n"
    f = build(registry, tmp_path, "a.py", code)
    assert "collections.py" in f.imports
    assert not any("as" in imp for imp in f.imports)
    assert not any(" " in imp for imp in f.imports)


def test_aliased_named_import_stores_original_uid(tmp_path, registry):
    code = "from pathlib import Path as P\n"
    f = build(registry, tmp_path, "a.py", code)
    # The alias is dropped; the import normalizes to the real symbol's UID candidate.
    assert "pathlib.py#Path" in f.imports
    assert not any("#P" in imp and "#Path" not in imp for imp in f.imports)


def test_wildcard_import(tmp_path, registry):
    code = "from math import *\n"
    f = build(registry, tmp_path, "a.py", code)
    assert "math.py.*" in f.imports


# ---------------------------------------------------------------------------
# Arity
# ---------------------------------------------------------------------------

def test_self_excluded_from_arity(tmp_path, registry):
    code = "class Foo:\n    def bar(self, x, y):\n        pass\n"
    build(registry, tmp_path, "foo.py", code)
    m = [x for x in registry.methods if x.name == "bar"][0]
    assert m.arity == 2


def test_cls_excluded_from_arity(tmp_path, registry):
    code = "class Foo:\n    @classmethod\n    def create(cls, name):\n        pass\n"
    build(registry, tmp_path, "foo.py", code)
    m = [x for x in registry.methods if x.name == "create"][0]
    assert m.arity == 1


def test_free_function_arity_unchanged(tmp_path, registry):
    code = "def helper(a, b, c):\n    pass\n"
    build(registry, tmp_path, "foo.py", code)
    m = [x for x in registry.methods if x.name == "helper"][0]
    assert m.arity == 3


# ---------------------------------------------------------------------------
# Local (same-file) dependency resolution
# ---------------------------------------------------------------------------

def test_local_free_function_call(tmp_path, registry):
    code = """
def helper():
    pass

def main():
    helper()
"""
    f = build(registry, tmp_path, "a.py", code)
    resolve_all(registry)
    main = [x for x in registry.methods if x.name == "main"][0]
    helper = [x for x in registry.methods if x.name == "helper"][0]
    assert helper in main.outbound_dependencies


def test_local_self_method_call(tmp_path, registry):
    code = """
class Calculator:
    def add(self, a, b):
        return a + b

    def compute(self, x, y):
        return self.add(x, y)
"""
    f = build(registry, tmp_path, "calc.py", code)
    resolve_all(registry)
    compute = [x for x in registry.methods if x.name == "compute"][0]
    add = [x for x in registry.methods if x.name == "add"][0]
    assert add in compute.outbound_dependencies


def test_local_cls_method_call(tmp_path, registry):
    code = """
class Repo:
    @classmethod
    def _connect(cls):
        pass

    @classmethod
    def open(cls):
        return cls._connect()
"""
    f = build(registry, tmp_path, "repo.py", code)
    resolve_all(registry)
    open_m = [x for x in registry.methods if x.name == "open"][0]
    connect = [x for x in registry.methods if x.name == "_connect"][0]
    assert connect in open_m.outbound_dependencies


# ---------------------------------------------------------------------------
# Cross-file dependency resolution (import-based)
# ---------------------------------------------------------------------------

def test_imported_class_instantiation(tmp_path, registry):
    models_code = """
class User:
    def __init__(self, name):
        self.name = name
"""
    service_code = """
from models import User

class UserService:
    def create(self, name):
        return User(name)
"""
    build(registry, tmp_path, "models.py", models_code)
    build(registry, tmp_path, "service.py", service_code)
    resolve_all(registry)

    create = [x for x in registry.methods if x.name == "create"][0]
    user_class = registry.get_struct_by_uid("models.py#User")
    assert user_class in create.outbound_dependencies


def test_imported_function_call(tmp_path, registry):
    utils_code = """
def format_currency(amount):
    return f"${amount:.2f}"
"""
    service_code = """
from utils import format_currency

def show_price(amount):
    return format_currency(amount)
"""
    build(registry, tmp_path, "utils.py", utils_code)
    build(registry, tmp_path, "service.py", service_code)
    resolve_all(registry)

    show = [x for x in registry.methods if x.name == "show_price"][0]
    fmt = [x for x in registry.methods if x.name == "format_currency"][0]
    assert fmt in show.outbound_dependencies


def test_method_call_on_imported_instance(tmp_path, registry):
    """obj.method() where obj is a param — resolved via import + method lookup."""
    models_code = """
class User:
    def get_display_name(self):
        return self.name
"""
    service_code = """
from models import User

class UserService:
    def get_display(self, user):
        return user.get_display_name()
"""
    build(registry, tmp_path, "models.py", models_code)
    build(registry, tmp_path, "service.py", service_code)
    resolve_all(registry)

    get_display = [x for x in registry.methods if x.name == "get_display"][0]
    get_display_name = [x for x in registry.methods if x.name == "get_display_name"][0]
    # May resolve exact or via fuzzy — check either set
    all_deps = get_display.outbound_dependencies | get_display.outbound_dependencies_fuzzy
    assert get_display_name in all_deps or get_display_name.parent in get_display.outbound_dependencies


# ---------------------------------------------------------------------------
# Alias normalization
# ---------------------------------------------------------------------------

def test_aliased_named_import_call_resolves(tmp_path, registry):
    models_code = """
class User:
    def greet(self):
        return "hi"
"""
    service_code = """
from models import User as U

class Service:
    def run(self):
        return U()
"""
    build(registry, tmp_path, "models.py", models_code)
    build(registry, tmp_path, "service.py", service_code)
    resolve_all(registry)

    run = [x for x in registry.methods if x.name == "run"][0]
    user_class = registry.get_struct_by_uid("models.py#User")
    assert user_class in run.outbound_dependencies


def test_aliased_module_import_call_resolves(tmp_path, registry):
    utils_code = """
def helper():
    pass
"""
    main_code = """
import utils as u

def main():
    u.helper()
"""
    build(registry, tmp_path, "utils.py", utils_code)
    build(registry, tmp_path, "main.py", main_code)
    resolve_all(registry)

    main = [x for x in registry.methods if x.name == "main"][0]
    helper = [x for x in registry.methods if x.name == "helper"][0]
    assert helper in main.outbound_dependencies


# ---------------------------------------------------------------------------
# Field type annotation resolution
# ---------------------------------------------------------------------------

def test_field_type_annotation_parsed(tmp_path, registry):
    code = """
class Client:
    def post(self, url):
        pass

class Service:
    client: Client

    def call(self):
        self.client.post("/api")
"""
    build(registry, tmp_path, "app.py", code)
    resolve_all(registry)

    call = [x for x in registry.methods if x.name == "call"][0]
    post = [x for x in registry.methods if x.name == "post"][0]
    assert post in call.outbound_dependencies


# ---------------------------------------------------------------------------
# Inheritance
# ---------------------------------------------------------------------------

def test_inherited_method_call_via_self(tmp_path, registry):
    code = """
class Base:
    def shared(self):
        pass

class Child(Base):
    def run(self):
        self.shared()
"""
    build(registry, tmp_path, "hierarchy.py", code)
    resolve_all(registry)

    run = [x for x in registry.methods if x.name == "run"][0]
    shared = [x for x in registry.methods if x.name == "shared"][0]
    assert shared in run.outbound_dependencies


def test_class_inheritance_dependency(tmp_path, registry):
    code = """
class Animal:
    def speak(self):
        pass

class Dog(Animal):
    pass
"""
    build(registry, tmp_path, "animals.py", code)
    resolve_all(registry)

    dog = registry.get_struct_by_uid("animals.py#Dog")
    animal = registry.get_struct_by_uid("animals.py#Animal")
    assert animal in dog.outbound_dependencies


# ---------------------------------------------------------------------------
# Relative imports
# ---------------------------------------------------------------------------

def test_relative_import_resolution(tmp_path, registry):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "models.py").write_text("class Item:\n    def describe(self):\n        pass\n")
    (pkg / "service.py").write_text("from .models import Item\n\nclass Svc:\n    def run(self):\n        return Item()\n")

    builder = PythonFileBuilder(registry)
    for p in [pkg / "models.py", pkg / "service.py"]:
        file_obj = builder.from_path(p)
        registry.add_struct(file_obj)

    resolve_all(registry)

    run = [x for x in registry.methods if x.name == "run"][0]
    item_class = registry.get_struct_by_uid("pkg/models.py#Item")
    assert item_class in run.outbound_dependencies


# ---------------------------------------------------------------------------
# Source-root anchoring of absolute imports
# ---------------------------------------------------------------------------

def build_at(registry, tmp_path, relpath, code):
    p = tmp_path / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(code)
    file_obj = PythonFileBuilder(registry).from_path(p)
    registry.add_struct(file_obj)
    return file_obj


def test_src_layout_absolute_import_resolves(tmp_path, registry):
    """`src/pkg/cmd.py` importing `pkg.a` anchors to the shared source root `src/`."""
    build_at(registry, tmp_path, "src/pkg/a.py", "class A:\n    pass\n")
    build_at(registry, tmp_path, "src/pkg/cmd.py",
             "from pkg.a import A\n\nclass Cmd:\n    def run(self):\n        return A()\n")
    resolve_all(registry)

    run = [x for x in registry.methods if x.name == "run"][0]
    a_class = registry.get_struct_by_uid("src/pkg/a.py#A")
    assert a_class in run.outbound_dependencies


def test_shadowed_anchor_falls_back_to_suffix_match(tmp_path, registry):
    """A directory in the importer's own path that shares the import's first segment must not
    hijack the anchor: `docs/pkg/example.py` importing `pkg.a` still resolves to `src/pkg/a.py`
    via the unanchored candidate's suffix match."""
    build_at(registry, tmp_path, "src/pkg/a.py", "class A:\n    pass\n")
    build_at(registry, tmp_path, "docs/pkg/example.py",
             "from pkg.a import A\n\nclass Example:\n    def run(self):\n        return A()\n")
    resolve_all(registry)

    run = [x for x in registry.methods if x.name == "run"][0]
    a_class = registry.get_struct_by_uid("src/pkg/a.py#A")
    assert a_class in run.outbound_dependencies


# ---------------------------------------------------------------------------
# Re-export fallback gating
# ---------------------------------------------------------------------------

def test_package_reexport_resolves_to_definition(tmp_path, registry):
    """`from pkg import Widget` where Widget lives in pkg/impl.py and is re-exported by
    pkg/__init__.py resolves through the unique-name fallback."""
    build_at(registry, tmp_path, "pkg/impl.py", "class Widget:\n    pass\n")
    build_at(registry, tmp_path, "pkg/__init__.py", "from pkg.impl import Widget\n")
    build_at(registry, tmp_path, "main.py",
             "from pkg import Widget\n\nclass App:\n    def run(self):\n        return Widget()\n")
    resolve_all(registry)

    run = [x for x in registry.methods if x.name == "run"][0]
    widget = registry.get_struct_by_uid("pkg/impl.py#Widget")
    assert widget in run.outbound_dependencies


def test_external_import_does_not_false_match_project_struct(tmp_path, registry):
    """`from loguru import logger` must not resolve to a same-named project struct: the
    unique-name fallback only fires when the import's module is itself a project file."""
    build_at(registry, tmp_path, "util.py", "def logger():\n    pass\n")
    build_at(registry, tmp_path, "main.py",
             "from loguru import logger\n\nclass App:\n    def run(self):\n        pass\n")
    resolve_all(registry)

    logger_fn = registry.get_struct_by_uid("util.py#logger(...)")
    app = registry.get_struct_by_uid("main.py#App")
    main_file = registry.get_struct_by_uid("main.py")
    assert logger_fn is not None
    assert logger_fn not in app.outbound_dependencies
    assert logger_fn not in main_file.outbound_dependencies


# ---------------------------------------------------------------------------
# Wildcard import
# ---------------------------------------------------------------------------

def test_wildcard_import_fuzzy_resolution(tmp_path, registry):
    models_code = """
class Foo:
    def do_it(self):
        pass
"""
    service_code = """
from models import *

class Bar:
    def run(self):
        Foo()
"""
    build(registry, tmp_path, "models.py", models_code)
    build(registry, tmp_path, "service.py", service_code)
    resolve_all(registry)

    run = [x for x in registry.methods if x.name == "run"][0]
    foo = registry.get_struct_by_uid("models.py#Foo")
    # Wildcard matches land in outbound_dependencies (if unambiguous) or fuzzy
    all_deps = run.outbound_dependencies | run.outbound_dependencies_fuzzy
    assert foo in all_deps
