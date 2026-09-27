#!/usr/bin/env python3

import json
from pathlib import Path
from collections import Counter

CLEAN_DIR = Path("swarnasky_clean")
PAGES_DIR = CLEAN_DIR / "pages"
OUTPUT_DIR = Path("swarnasky_chunks")

OUTPUT_FILE = OUTPUT_DIR / "chunks.jsonl"
REPORT_FILE = OUTPUT_DIR / "chunking_report.json"


def clean_text(text):
    if not isinstance(text, str):
        return ""
    return " ".join(text.split()).strip()


def main():
    if not PAGES_DIR.exists():
        raise SystemExit(f"ERROR: {PAGES_DIR} not found")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    chunks = []
    pages_processed = 0
    sections_processed = 0
    skipped_sections = 0
    word_counts = []

    content_files = sorted(PAGES_DIR.glob("*/content.json"))

    for content_file in content_files:
        try:
            data = json.loads(content_file.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"ERROR reading {content_file}: {exc}")
            continue

        if not data.get("accepted"):
            continue

        page_id = data.get("page_id") or content_file.parent.name
        url = data.get("final_url") or data.get("url") or ""
        title = clean_text(data.get("title", ""))

        sections = data.get("sections", [])
        if not isinstance(sections, list):
            continue

        pages_processed += 1
        page_chunk_count = 0

        for section in sections:
            if not isinstance(section, dict):
                skipped_sections += 1
                continue

            # The cleaner already created RAG-ready text.
            text = clean_text(section.get("rag_text", ""))

            # Fallback if rag_text is missing.
            if not text:
                paragraphs = section.get("paragraphs", [])
                list_items = section.get("list_items", [])

                parts = []
                if isinstance(paragraphs, list):
                    parts.extend(clean_text(x) for x in paragraphs if clean_text(x))
                if isinstance(list_items, list):
                    parts.extend(clean_text(x) for x in list_items if clean_text(x))

                text = "\n".join(parts)

            if not text:
                skipped_sections += 1
                continue

            section_id = section.get("section_id", page_chunk_count)
            heading = clean_text(section.get("heading", ""))
            level = section.get("level")
            section_path = section.get("section_path", [])

            if not isinstance(section_path, list):
                section_path = []

            section_path = [clean_text(x) for x in section_path if clean_text(x)]

            word_count = len(text.split())

            chunk = {
                "chunk_id": f"{page_id}_{section_id}",
                "page_id": page_id,
                "url": url,
                "page_title": title,
                "section_id": section_id,
                "heading": heading,
                "level": level,
                "section_path": section_path,
                "text": text,
                "word_count": word_count,
                "source_file": str(content_file),
            }

            chunks.append(chunk)
            sections_processed += 1
            page_chunk_count += 1
            word_counts.append(word_count)

        print(
            f"CHUNKED  {page_id}  "
            f"{page_chunk_count:2d} chunks  "
            f"{sum(c['word_count'] for c in chunks if c['page_id'] == page_id):4d} words  "
            f"'{title}'"
        )

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    report = {
        "chunking_version": "2.0",
        "strategy": "one cleaned section per chunk",
        "source": "swarnasky_clean/pages/*/content.json",
        "pages_processed": pages_processed,
        "sections_processed": sections_processed,
        "skipped_sections": skipped_sections,
        "total_chunks": len(chunks),
        "total_words": sum(word_counts),
        "min_chunk_words": min(word_counts) if word_counts else 0,
        "max_chunk_words": max(word_counts) if word_counts else 0,
        "average_chunk_words": (
            round(sum(word_counts) / len(word_counts), 2)
            if word_counts else 0
        ),
        "note": (
            "No rewriting, summarization, or semantic splitting. "
            "The cleaner's existing rag_text is used as the primary chunk text."
        ),
    }

    REPORT_FILE.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 48)
    print("CHUNKING COMPLETE")
    print("=" * 48)
    print(f"Pages processed : {pages_processed}")
    print(f"Total chunks    : {len(chunks)}")
    print(f"Total words     : {sum(word_counts)}")
    print(f"Output          : {OUTPUT_FILE}")
    print(f"Report          : {REPORT_FILE}")


if __name__ == "__main__":
    main()
