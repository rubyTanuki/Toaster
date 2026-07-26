from __future__ import annotations
import platform
import shutil
import urllib.request
from functools import cached_property
from pathlib import Path
from loguru import logger
import numpy as np
from .base import EmbeddingStrategy

_CACHE_DIR = Path.home() / ".cache" / "tostr" / "models" / "all-MiniLM-L6-v2"
_HF_BASE = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main"

# Architecture-specific quantized ONNX builds (~23 MB each, vs ~90 MB unquantized).
# x86-64 maps to the AVX2 build rather than an AVX-512 variant: AVX2 is present on
# effectively all modern x86 CPUs, whereas AVX-512 is not and would fault at inference.
_FALLBACK_MODEL = "model.onnx"
_ARCH_MODELS = {
    "arm64": "model_qint8_arm64.onnx",
    "aarch64": "model_qint8_arm64.onnx",
    "x86_64": "model_quint8_avx2.onnx",
    "amd64": "model_quint8_avx2.onnx",
}
_TOKENIZER_FILENAME = "tokenizer.json"
# Minimum acceptable sizes catch truncated downloads before onnxruntime tries to parse them.
_MODEL_MIN_SIZE = 15 * 1024 * 1024
_TOKENIZER_MIN_SIZE = 100 * 1024
_DOWNLOAD_HEADERS = {"User-Agent": "tostr/1.0"}

class OnnxEmbeddingStrategy(EmbeddingStrategy):
    def __init__(self, batch_size: int = 32, batch_timeout: float = 1.5):
        super().__init__(batch_size=batch_size, batch_timeout=batch_timeout)

        self.model_dir = _CACHE_DIR
        self.vocab_path = str(self.model_dir / _TOKENIZER_FILENAME)

        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._ensure_asset(_TOKENIZER_FILENAME, f"{_HF_BASE}/{_TOKENIZER_FILENAME}")
        self.tokenizer = Tokenizer.from_file(self.vocab_path)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

        self.onnx_path, self.session = self._load_session(ort)

    @property
    def dimensions(self) -> int:
        return 384

    @cached_property
    def model_filename(self) -> str:
        """The quantized ONNX build matching the host CPU, else the portable default."""
        return _ARCH_MODELS.get(platform.machine().lower(), _FALLBACK_MODEL)

    @cached_property
    def _model_candidates(self) -> list[str]:
        """Models to try in order: architecture-specific first, portable default last.

        dict.fromkeys dedupes while preserving order (the arch model may already be the
        fallback on unrecognized CPUs).
        """
        return list(dict.fromkeys([self.model_filename, _FALLBACK_MODEL]))

    def _load_session(self, ort):
        """Return (path, session) for the first candidate that downloads and initializes.

        A quantized build can 404 or fail to run on an unexpected CPU. Rather than leave
        the pipeline with no model at all, degrade to the portable model.onnx before
        giving up entirely.
        """
        errors: list[str] = []
        for filename in self._model_candidates:
            try:
                path = self._ensure_model(filename)
                session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            except Exception as e:
                logger.warning(f"Embedding model '{filename}' unavailable ({e}); trying next candidate.")
                errors.append(f"{filename}: {e}")
                continue
            if filename != self.model_filename:
                logger.warning(f"Falling back to portable embedding model '{filename}'.")
            return path, session

        raise RuntimeError(
            "Could not initialize any embedding model. Tried:\n  " + "\n  ".join(errors)
        )

    @staticmethod
    def _min_size(filename: str) -> int:
        return _TOKENIZER_MIN_SIZE if filename == _TOKENIZER_FILENAME else _MODEL_MIN_SIZE

    def _asset_is_valid(self, filename: str) -> bool:
        dest = self.model_dir / filename
        return dest.exists() and dest.stat().st_size >= self._min_size(filename)

    def _ensure_model(self, filename: str) -> str:
        """Ensure a specific ONNX build is cached; returns its local path."""
        self._ensure_asset(filename, f"{_HF_BASE}/onnx/{filename}")
        return str(self.model_dir / filename)

    def _ensure_asset(self, filename: str, url: str):
        """Download one asset to the cache if missing/truncated. Raises on failure."""
        if self._asset_is_valid(filename):
            return

        self.model_dir.mkdir(parents=True, exist_ok=True)
        dest = self.model_dir / filename
        dest.unlink(missing_ok=True)  # remove any partial file from a prior interrupted download
        tmp = dest.with_suffix(".tmp")
        logger.info(f"Downloading embedding asset {filename} from Hugging Face Hub ...")
        try:
            req = urllib.request.Request(url, headers=_DOWNLOAD_HEADERS)
            with urllib.request.urlopen(req, timeout=60) as resp:
                with open(tmp, "wb") as f:
                    shutil.copyfileobj(resp, f)
            tmp.rename(dest)
        except Exception as e:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to download {filename} from Hugging Face Hub.\n"
                f"URL: {url}\nError: {e}"
            ) from e

        actual = dest.stat().st_size
        min_size = self._min_size(filename)
        if actual < min_size:
            dest.unlink(missing_ok=True)
            raise RuntimeError(
                f"Download of {filename} appears incomplete ({actual / 1024 / 1024:.1f} MB). "
                f"Expected at least {min_size // 1024 // 1024} MB. "
                f"Please try again."
            )

    def _execute_onnx(self, texts: list[str]) -> list[list[float]]:
        """Executes compiled math graph using tokenizers and ONNX runtime layers."""
        # Clean out empty calls early
        if not texts:
            return []
            
        # Fast Rust Tokenization (Runs in <1ms)
        encoded = self.tokenizer.encode_batch(texts)
        
        # Convert input mappings to raw NumPy containers
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        
        # MiniLM expects a token type index layer (usually zeros)
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)
        
        # Prepare execution payload mapping to ONNX variable expectations
        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        }
        
        # Execute compiled forward-pass matrices
        ort_outputs = self.session.run(None, ort_inputs)
        
        # Output indices depend on configuration (Typically index 0 is token embeddings)
        token_embeddings = ort_outputs[0] 
        
        # Perform Mean Pooling over the attention mask to compute structural sequence tokens
        input_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(float)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        
        # Calculate centroids and normalize vector profiles to Euclidean unit length
        pooled = sum_embeddings / sum_mask
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        normalized_embeddings = pooled / np.clip(norms, a_min=1e-9, a_max=None)
        
        return normalized_embeddings.tolist()

    def embed_batch(self, descriptions: list[str]) -> list[list[float]]:
        return self._execute_onnx(descriptions)

    def embed_query(self, query: str) -> list[float]:
        return self._execute_onnx([query])[0]