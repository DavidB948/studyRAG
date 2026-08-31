"""File -> RawDoc. Mechanical extraction and cleaning only; no structure parsing.

Cleaning lives here rather than in the chunker because it is format-level
repair with no design judgement in it — the chunker should never see base64.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from studyrag.config import settings
from studyrag.types import DocType, RawDoc

log = logging.getLogger(__name__)

TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
SUPPORTED = TEXT_SUFFIXES | {".pdf"}

# LaTeXiT stores each formula's source as an invisible text blob. On a Beamer
# deck these are ~92% of extracted characters. They decode to worked numeric
# examples rather than named equations, so they are replaced, not recovered.
LATEXIT = re.compile(r"<latexit.*?</latexit>", re.DOTALL)

# Stray base64 that survives the tag-based match above.
LONG_B64 = re.compile(r"[A-Za-z0-9+/]{60,}={0,2}")

# Control characters that pdf extraction leaks and Postgres `text` cannot store: NUL
# outright rejects the insert, the rest are invisible noise inside an embedding.
# Tab and newline are kept — they carry the line structure the chunker reads.
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def infer_doc_type(doc_path: str) -> DocType:
    name = doc_path.lower()
    if "solution" in name or "answer" in name:
        return "tutorial_soln"
    if "tutorial" in name or "problem" in name:
        return "tutorial_q"
    if "slide" in name or "lecture" in name:
        return "slides"
    return "notes"


def clean(text: str) -> str:
    text = LATEXIT.sub(" [FORMULA] ", text)
    text = LONG_B64.sub(" [FORMULA] ", text)
    text = text.replace("\r\n", "\n").replace("\xa0", " ")
    text = CONTROL_CHARS.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _pdf_pages(path: Path) -> list[str]:
    from pypdf import PdfReader

    return [(page.extract_text() or "") for page in PdfReader(str(path)).pages]


def identify(path: str | Path) -> tuple[str, str, str]:
    """Return (course, doc_path, content_hash) without extracting any text.

    Split out from `load` so ingest can answer "have I seen this file?" for the
    cost of a read and a hash. Extraction is the expensive part, and re-parsing
    every PDF in the corpus to skip it is wasted work when one file is added.

    Hashed before cleaning: this identifies the file as delivered, so changing
    the cleaning rules must not silently change every document's identity.
    """
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED:
        raise ValueError(f"Unsupported file type {path.suffix}; expected one of {SUPPORTED}")

    root = settings.corpus_root.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{path} is outside the corpus root {root}")

    rel = resolved.relative_to(root)
    if len(rel.parts) < 2:
        raise ValueError(f"{rel} has no course directory; expected <COURSE>/<file>")

    content_hash = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return rel.parts[0], str(Path(*rel.parts[1:])), content_hash


def load(path: str | Path, doc_type: DocType | None = None) -> RawDoc:
    path = Path(path)
    course, doc_path, content_hash = identify(path)
    data = path.read_bytes()

    raw = _pdf_pages(path) if path.suffix.lower() == ".pdf" else [data.decode(errors="replace")]
    pages = [clean(p) for p in raw]

    if not any(pages):
        raise ValueError(f"{course}/{doc_path} produced no text — likely a scanned PDF needing OCR")

    return RawDoc(
        course=course,
        doc_path=doc_path,
        doc_type=doc_type or infer_doc_type(doc_path),
        content_hash=content_hash,
        pages=pages,
    )


def discover(root: Path | None = None) -> list[Path]:
    """Every ingestable file under the root.

    `is_file` and the dotfile skip are not paranoia: macOS writes `._name.pdf`
    AppleDouble sidecars, and a directory can be named `notes.md`.
    """
    root = root or settings.corpus_root
    if not root.exists():
        log.error("corpus root %s does not exist", root.resolve())
        return []

    supported: list[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name.startswith("."):
            continue
        if p.suffix.lower() in SUPPORTED:
            supported.append(p)
        else:
            # Warn rather than ignore: a file the user deliberately put in the corpus
            # and never sees again is worse than a noisy line. Scanned PDFs get their
            # own message later, from load().
            log.warning(
                "unsupported file type %s, skipping %s (supported: %s)",
                p.suffix or "<none>",
                p.relative_to(root),
                ", ".join(sorted(SUPPORTED)),
            )
    return supported
