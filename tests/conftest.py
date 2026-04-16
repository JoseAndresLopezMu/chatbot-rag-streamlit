"""
Global test setup for chatbot-rag-streamlit.

The project depends on a heavy ML stack (chromadb, langchain, easyocr, torch…)
that is too large to install in CI.  We intercept all of those packages in
sys.modules *before* any project module is imported so the real packages are
never loaded.  Only lightweight stdlib + loguru + tqdm are needed.
"""

import sys
from unittest.mock import MagicMock

_HEAVY_MODULES = [
    "chromadb",
    "langchain_community",
    "langchain_community.embeddings",
    "langchain_community.vectorstores",
    "langchain_core",
    "langchain_core.documents",
    "langchain",
    "langchain_chroma",
    "langchain_ollama",
    "langchain_text_splitters",
    "pymupdf4llm",
    "cache_to_disk",
    "chonkie",
    "chonkie.chunker",
    "chonkie.chunker.base",
    "chonkie.utils",
    "easyocr",
    "fitz",
    "unstructured",
    "pdfminer",
]

for _mod in _HEAVY_MODULES:
    sys.modules.setdefault(_mod, MagicMock())

# cache_to_disk decorator must return the original function unchanged
# so that monkeypatch.setattr works on the real function objects.
_cache_mod = MagicMock()
_cache_mod.cache_to_disk = lambda *args, **kwargs: (lambda fn: fn)
sys.modules["cache_to_disk"] = _cache_mod

# langchain_core.documents.Document needs to behave like a simple dataclass
from dataclasses import dataclass  # noqa: E402


@dataclass
class _FakeDocument:
    page_content: str
    metadata: dict


_lc_docs = MagicMock()
_lc_docs.Document = _FakeDocument
sys.modules["langchain_core.documents"] = _lc_docs
