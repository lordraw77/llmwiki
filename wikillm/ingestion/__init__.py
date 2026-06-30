"""Document ingestion: format detection, text extraction and chunking.

The public surface is intentionally small:

* :func:`wikillm.ingestion.parsers.parse_bytes` /
  :func:`wikillm.ingestion.parsers.parse_path` extract plain text from a source.
* :func:`wikillm.ingestion.chunking.chunk_text` splits that text into
  embeddable chunks.
"""

from __future__ import annotations

from wikillm.ingestion.chunking import chunk_text
from wikillm.ingestion.parsers import (
    SUPPORTED_EXTENSIONS,
    ParseError,
    UnsupportedFormatError,
    detect_type,
    iter_supported_files,
    parse_bytes,
    parse_path,
)

__all__ = [
    "chunk_text",
    "SUPPORTED_EXTENSIONS",
    "ParseError",
    "UnsupportedFormatError",
    "detect_type",
    "iter_supported_files",
    "parse_bytes",
    "parse_path",
]
