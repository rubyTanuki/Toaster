from __future__ import annotations
import tomllib
from pathlib import Path
from typing import Dict, List
import pathspec

from loguru import logger

# Bundled language ignore templates live at src/tostr/languages/<lang>/default.tostrignore.
# config.py sits at the package root, so languages/ is one parent up.
_LANGUAGES_DIR = Path(__file__).resolve().parents[0] / "languages"


def _languages_for(language: str) -> List[str]:
    """Resolve a config language value to the concrete language(s) whose ignore defaults apply.
    "auto" expands to every supported language; an explicit value is used as-is."""
    if language == "auto":
        from tostr.core.providers import LanguageProvider
        return list(LanguageProvider.language_map)
    return [language]


def language_default_ignore_lines(language: str) -> List[str]:
    """Bundled default-ignore lines for the active language(s).

    These are the in-code default layer (§5): applied by ProjectConfig only when the project has no
    authored `.tostrignore`. Materialized to disk by `init` so users can see/edit them."""
    lines: List[str] = []
    for lang in _languages_for(language):
        template = _LANGUAGES_DIR / lang / "default.tostrignore"
        if template.exists():
            lines.extend(template.read_text().splitlines())
    return lines


def default_ignore_text(language: str) -> str:
    """The materialized `.tostrignore` body `init` writes: each active language's template,
    concatenated under a header so the source of each block is obvious."""
    chunks: List[str] = []
    for lang in _languages_for(language):
        template = _LANGUAGES_DIR / lang / "default.tostrignore"
        if template.exists():
            chunks.append(f"# --- {lang} defaults ---\n{template.read_text().rstrip()}")
    return "\n\n".join(chunks) + "\n"


class ProjectConfig:
    HARDCODED_IGNORES = [
        '.DS_Store',
        '*.exe',
        '*.bin',
        '*.dll',
        '*.so',
        '*.dylib',
        '*.pyc',
        '*.pyo',
        '*.pyd',
        '__pycache__/',
        '.git/',
        '.svn/',
        '.hg/',
        '.tostr/',
        '.tostrignore',
        '*.json',
        '*.md',
        '*.kts',
        '*.properties',
        '*.xml',
        '*.gradle'
    ]

    def __init__(self, project_path: Path, overrides: Dict | None = None):
        self.project_path = project_path
        # Per-invocation overrides (e.g. CLI --language); win over the on-disk tostr.toml. Set
        # before the ignore spec is built so the default-ignore layer uses the resolved language.
        self.overrides = overrides or {}
        self.toml_config = self._init_toml_config(project_path)
        self.ignore_rules = self._init_path_ignore(project_path)
        self.hardcoded_rules = pathspec.PathSpec.from_lines('gitignore', self.HARDCODED_IGNORES)

    @property
    def language(self) -> str:
        # "auto" parses every supported language, routed per-file by extension.
        if self.overrides.get("language"):
            return self.overrides["language"]
        return self.toml_config.get("project", {}).get("language", "auto")

    @property
    def llm_strategy(self) -> str:
        # Resolution order: per-invocation override (--llm) > tostr.toml [llm].strategy > "gemini".
        # "none" disables LLM description generation (equivalent to --no-llm).
        if self.overrides.get("llm"):
            return self.overrides["llm"]
        return self.toml_config.get("llm", {}).get("strategy", "gemini")

    @property
    def llm_options(self) -> Dict:
        # Provider-specific settings from the [llm] table (e.g. model, base_url), excluding the
        # strategy selector itself. Passed through to the chosen strategy's constructor.
        options = dict(self.toml_config.get("llm", {}))
        options.pop("strategy", None)
        return options

    def _init_toml_config(self, project_path: Path) -> Dict:
        # Authored config now lives at the project root (tostr.toml), not inside generated .tostr/.
        toml_path = project_path / "tostr.toml"
        if toml_path.exists():
            with open(toml_path, 'rb') as f:
                config = tomllib.load(f)
            logger.debug(f"Loaded configuration from {toml_path}")
            return config
        logger.debug("No tostr.toml found, using defaults.")
        return {}

    def _init_path_ignore(self, project_path: Path) -> pathspec.PathSpec:
        # An authored .tostrignore is the single source of truth for non-hardcoded ignores: when it
        # exists, the bundled language defaults are turned off entirely (init materializes the
        # defaults into the file, so nothing is lost). Otherwise fall back to the default layer.
        ignore_path = project_path / ".tostrignore"
        if ignore_path.exists():
            with open(ignore_path, 'r') as f:
                return pathspec.PathSpec.from_lines('gitignore', f)
        return pathspec.PathSpec.from_lines('gitignore', language_default_ignore_lines(self.language))

    def is_ignored(self, file_path: Path) -> bool:
        if not self.project_path:
            return False

        # 1. Convert to a POSIX string relative to the project root.
        # Relative inputs are anchored to project_path, NOT the process CWD
        # (which is what Path.absolute() would otherwise use for a relative path).
        anchored = file_path if file_path.is_absolute() else self.project_path / file_path
        try:
            # We prefer absolute() over resolve() to avoid following symlinks
            # that point outside the project tree during logical traversal.
            relative_path = anchored.absolute().relative_to(self.project_path.absolute()).as_posix()
        except ValueError:
            # Fallback to resolve() if they are in different places but logically linked
            try:
                relative_path = anchored.resolve().relative_to(self.project_path.resolve()).as_posix()
            except ValueError:
                # If the file is outside the project root, we should probably ignore it
                return True

        if relative_path == ".":
            return False

        # If it's a directory, append a slash so directory-only rules (like `dist/`) can match it
        if file_path.is_dir() and not relative_path.endswith('/'):
            relative_path += '/'

        # Check hardcoded rules first
        if self.hardcoded_rules.match_file(relative_path):
            return True

        return self.ignore_rules.match_file(relative_path)
