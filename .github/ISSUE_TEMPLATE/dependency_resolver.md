---
name: "Dependency Resolution Implementation"
about: "Standard template for assigning expanding the support of the dependency resolver to languages which currently only support parsing."
title: "FEAT: "
labels: 'enhancement' 'language support'
assignees: ''
---

**Context:**
Dependency resolution is a vastly more complicated issue to handle than struct extraction and AST building, making implementing it a difficult task of its own apart from the initial parsing support. [LANGUAGE] already has parsing support as was added in [LANGUAGE PARSER ISSUE #], but still needs dependency resolution to be up to speed with the full feature suite available with the java parser.

**Task Details:**

Implement full [LANGUAGE] dependency parsing and resolution.
1. Add import parsing and cleaning to `languages/[LANGUAGE]/builders.py` for the file factory.
2. Integrate the dependency resolution in the language's `resolver.py` (`src/tostr/languages/<lang>/resolver.py`).
3. Update the unit test suite to ensure dependencies are resolved properly. Typically we just use llms for this but due to the complexity of dependency resolution, this should be looked at with extra care. A comprehensive list of all ways [LANGUAGE] structs can depend on another should be enumerated, and all that could reasonably be resolved by tostr should be asserted.


**Testing Requirements:**
- Ensure all tests in `tests/` pass locally.
