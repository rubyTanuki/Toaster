from __future__ import annotations
import inspect
import os
from typing import Optional
from importlib import import_module

class LanguageProvider:
    # Maps language string to (package, builder_class, resolver_class)
    language_map = {
        "java": ("tostr.languages.java.builders", "JavaBuilder", "tostr.languages.java.resolver.JavaDependencyResolver"),
        "python": ("tostr.languages.python.builders", "PythonBuilder", "tostr.languages.python.resolver.PythonDependencyResolver"),
        "html": ("tostr.languages.html.builders", "HtmlBuilder", "tostr.core.resolver.BaseDependencyResolver"),
    }

    # Maps file extension to language key in language_map.
    extension_map = {
        ".java": "java",
        ".py": "python",
        ".html": "html",
        ".htm": "html",
    }

    @classmethod
    def language_for_extension(cls, ext: str) -> Optional[str]:
        """Resolves a file extension (e.g. '.py') to its language key, or None if unsupported."""
        return cls.extension_map.get(ext.lower()) if ext else None

    @classmethod
    def _language_enabled(cls, registry: "Registry", lang: str) -> bool:
        """In single-language mode (config language != 'auto'), only the configured language is
        parsed. In 'auto' mode every supported language is accepted and routed per-file."""
        configured = (registry.language or "auto").lower()
        return configured == "auto" or configured == lang

    @classmethod
    def get_builder(cls, registry: "Registry", ext: str) -> Optional["BaseBuilder"]:
        """Returns a builder for the file's extension, or None if the extension is
        unsupported or excluded by a single-language configuration."""
        lang = cls.language_for_extension(ext)
        if not lang or not cls._language_enabled(registry, lang):
            return None

        package, builder_name, _ = cls.language_map[lang]
        module = import_module(package)
        builder_class = getattr(module, builder_name)
        return builder_class(registry)

    @classmethod
    def get_resolver(cls, registry: "Registry", ext: str) -> "BaseDependencyResolver":
        """Returns the dependency resolver for the file's extension. Falls back to the
        base (no-op) resolver for extensions without language-specific resolution."""
        lang = cls.language_for_extension(ext)
        if lang and lang in cls.language_map:
            _, _, resolver_path = cls.language_map[lang]
            module_path, class_name = resolver_path.rsplit(".", 1)
            module = import_module(module_path)
            resolver_class = getattr(module, class_name)
            return resolver_class(registry)

        from tostr.core.resolver import BaseDependencyResolver
        return BaseDependencyResolver(registry)


class LLMProvider:
    # Maps an [llm].strategy value to the (module, class) implementing it. Each strategy's
    # constructor takes api_key/model_name plus provider-specific kwargs; build_client() sources
    # those from the [llm] config table (and the environment, for credentials).
    strategy_map = {
        "gemini": ("tostr.semantic.llm", "GeminiStrategy"),
        "ollama": ("tostr.semantic.llm", "OllamaStrategy"),
    }

    @classmethod
    def build_client(cls, config: "ProjectConfig", progress_tracker: "ProgressTracker" = None):
        """Construct the LLMClient for the resolved strategy, or None when LLM generation is off.

        Strategy resolution lives in ProjectConfig.llm_strategy (the `--llm` override beats
        tostr.toml's [llm].strategy, which beats the "gemini" default). The sentinel "none"
        disables description generation (equivalent to --no-llm)."""
        from tostr.semantic.llm import LLMClient

        name = (config.llm_strategy or "gemini").lower()
        if name == "none":
            return None

        strategy = cls._build_strategy(name, config.llm_options)
        return LLMClient(strategy=strategy, progress_tracker=progress_tracker)

    @classmethod
    def _build_strategy(cls, name: str, options: dict) -> "LLMStrategy":
        """Instantiate the named strategy, supplying credentials and the recognized [llm] options."""
        if name not in cls.strategy_map:
            from tostr.exceptions import ConfigError
            raise ConfigError(
                f"Unknown LLM strategy '{name}'. Set [llm].strategy in tostr.toml or pass --llm "
                f"with one of: {', '.join(sorted(cls.strategy_map))} (or 'none' to disable)."
            )

        module_path, class_name = cls.strategy_map[name]
        strategy_class = getattr(import_module(module_path), class_name)

        # The config key 'model' maps to every strategy's 'model_name' constructor param.
        kwargs = dict(options)
        if "model" in kwargs:
            kwargs.setdefault("model_name", kwargs.pop("model"))

        # Drop keys the chosen strategy doesn't accept so a stray [llm] option (e.g. base_url
        # under gemini) is ignored rather than raising a TypeError.
        accepted = set(inspect.signature(strategy_class.__init__).parameters)
        kwargs = {k: v for k, v in kwargs.items() if k in accepted}

        if name == "gemini":
            kwargs["api_key"] = cls._require_gemini_key()

        return strategy_class(**kwargs)

    @staticmethod
    def _require_gemini_key() -> str:
        from tostr.exceptions import APIKeyError
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise APIKeyError(
                "GEMINI_API_KEY is not set. Either set it, configure a different LLM binding via "
                "[llm].strategy in tostr.toml (e.g. 'ollama'), pass --llm, or use --no-llm to skip "
                "descriptions entirely."
            )
        return api_key
