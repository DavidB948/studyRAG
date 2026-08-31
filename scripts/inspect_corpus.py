"""Corpus inspection — look at the data before designing the chunker.

Prints what heading conventions actually appear under data/raw, so heading
detection is designed against reality rather than guesses. Read-only: no
database, no embedding, no writes, so it stays cheap enough to re-run often.

Usage:  uv run python scripts/inspect_corpus.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CORPUS = Path("data/raw")
TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
SUPPORTED = TEXT_SUFFIXES | {".pdf"}

PATTERNS = {
    "markdown_atx":    re.compile(r"^#{1,6}\s+\S"),            # ## Heading
    "markdown_setext": re.compile(r"^[=-]{3,}\s*$"),           # underlined heading
    "numbered":        re.compile(r"^\d+(\.\d+)*[.)]?\s+\S"),  # 4.2 Dynamic Programming
    "all_caps":        re.compile(r"^[A-Z][A-Z0-9 \-&/(),.]{4,60}$"),
    "keyword":         re.compile(r"^(chapter|lecture|week|topic|section|part)\b", re.I),
    "bullet":          re.compile(r"^\s*[-*•]\s+\S"),
    "page_marker":     re.compile(r"^<!-- page \d+ -->$"),
}

# LaTeXiT embeds the source of every formula image as an invisible text blob.
# On a Beamer deck these can be >90% of the extracted characters, which would
# swamp both the pattern counts and the raw sample below.
LATEXIT = re.compile(r"<latexit.*?</latexit>", re.S)


def read_text(path: Path) -> str:
    if path.suffix.lower() in TEXT_SUFFIXES:
        return path.read_text(errors="replace")
    from pypdf import PdfReader

    out = []
    for n, page in enumerate(PdfReader(str(path)).pages, start=1):
        t = (page.extract_text() or "").strip()
        if t:
            out.append(f"<!-- page {n} -->\n{t}")
    return "\n\n".join(out)


def inspect(path: Path) -> None:
    rel = path.relative_to(CORPUS)
    try:
        raw = read_text(path)
    except Exception as exc:
        print(f"\n{rel}\n  FAILED: {type(exc).__name__}: {exc}")
        return

    text = LATEXIT.sub(" [FORMULA] ", raw)
    lines = text.splitlines()

    print(f"\n{rel}")
    print(f"  {len(raw):,} chars raw -> {len(text):,} after stripping latexit"
          f" | {len(text.split()):,} words | {len(lines):,} lines")

    if not text.strip():
        print("  EMPTY — almost certainly a scanned PDF needing OCR")
        return

    hits = {}
    for name, pat in PATTERNS.items():
        matched = [ln.strip() for ln in lines if pat.match(ln.strip())]
        if matched:
            hits[name] = matched

    if not hits:
        print("  NO heading patterns matched — structure must come from position, not typography")
    for name, matched in sorted(hits.items(), key=lambda kv: -len(kv[1])):
        print(f"  {name}: {len(matched)}")
        for ln in matched[:4]:
            print(f"      {ln[:88]}")

    blank = sum(1 for ln in lines if not ln.strip()) / max(len(lines), 1)
    print(f"  blank lines: {blank:.0%}  (very low in a PDF usually means paragraph breaks were lost)")

    print("  --- first 15 non-empty lines ---")
    shown = 0
    for ln in lines:
        if ln.strip():
            print(f"      {ln[:88]}")
            shown += 1
            if shown == 15:
                break


def main() -> int:
    if not CORPUS.exists():
        print(f"{CORPUS} does not exist.")
        return 1

    files = sorted(p for p in CORPUS.rglob("*") if p.suffix.lower() in SUPPORTED)
    if not files:
        print(f"No supported files under {CORPUS}. Add some under data/raw/<COURSE>/.")
        return 1

    # `course` is NOT NULL and comes from the first path segment, so files sitting
    # loose at the corpus root would fail ingest. Surface that here instead.
    courses = sorted({
        p.relative_to(CORPUS).parts[0]
        for p in files if len(p.relative_to(CORPUS).parts) > 1
    })
    loose = [str(p.relative_to(CORPUS)) for p in files if len(p.relative_to(CORPUS).parts) == 1]

    print(f"{len(files)} file(s) | courses: {courses or 'NONE'}")
    if loose:
        print(f"  WARNING: no course directory for: {loose}")

    for f in files:
        inspect(f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
