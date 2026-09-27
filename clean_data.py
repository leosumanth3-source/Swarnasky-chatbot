from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Comment


# ============================================================
# Swarnasky Step 2
# Production-oriented content cleaning
#
# Input:
#   swarnasky_raw/pages/<page-id>/
#       rendered.html
#       raw.html
#       metadata.json
#
# Output:
#   swarnasky_clean/
#       pages/<page-id>/content.json
#       clean_manifest.jsonl
#       rejected_pages.jsonl
#       cleaning_report.json
#
# This step DOES NOT:
#   - summarize facts
#   - invent/rewrite website claims
#   - create embeddings
#   - write to Qdrant
#
# It extracts meaningful source content while preserving:
#   - page URL
#   - title
#   - section hierarchy
#   - paragraphs
#   - lists
#   - tables
#   - useful links
#   - source hashes
# ============================================================


INPUT_DIR = Path("swarnasky_raw")
OUTPUT_DIR = Path("swarnasky_clean")

MIN_PAGE_TEXT = 80
MIN_SECTION_TEXT = 25

# Elements that are almost always non-knowledge/UI content.
REMOVE_TAGS = {
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "canvas",
    "iframe",
    "object",
    "embed",
    "source",
    "track",
}

# Common boilerplate/UI class/id words. We use these conservatively.
NOISE_WORDS = {
    "cookie",
    "consent",
    "gdpr",
    "privacy-banner",
    "cookie-banner",
    "cookie-consent",
    "newsletter",
    "subscribe",
    "popup",
    "modal",
    "breadcrumb",
    "breadcrumbs",
    "social-share",
    "share-buttons",
    "sharethis",
    "related-posts",
    "related-content",
    "comments",
    "comment-form",
    "search-form",
    "site-search",
    "pagination",
    "pager",
    "advert",
    "advertisement",
    "ads",
    "promo-popup",
    "chat-widget",
    "live-chat",
}

# Text that is usually UI rather than knowledge.
NOISE_EXACT = {
    "menu",
    "search",
    "close",
    "read more",
    "learn more",
    "click here",
    "submit",
    "send",
    "next",
    "previous",
    "back",
    "home",
    "accept",
    "accept all",
    "reject all",
    "manage preferences",
    "subscribe",
    "sign up",
    "log in",
    "login",
}

# Repeated short site-wide strings are often navigation/footer boilerplate.
REPEATED_BOILERPLATE_MIN_LENGTH = 12
REPEATED_BOILERPLATE_MAX_LENGTH = 180
REPEATED_BOILERPLATE_MIN_PAGES = 3

CLEANING_VERSION = "1.1"


def clean_space(text: str) -> str:
    """Normalize whitespace without changing the words."""
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_for_comparison(text: str) -> str:
    """Normalization used only to detect duplicates."""
    text = clean_space(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def element_noise_score(tag) -> int:
    """
    Score an element for likely UI/boilerplate noise.

    We intentionally use a score instead of one aggressive rule.
    This reduces the chance of deleting real content just because
    a CMS happened to use a generic class name.
    """
    score = 0

    identity = " ".join(
        [
            str(tag.get("id", "")),
            " ".join(tag.get("class", []) if isinstance(tag.get("class"), list)
                     else [str(tag.get("class", ""))]),
            str(tag.get("role", "")),
            str(tag.get("aria-label", "")),
        ]
    ).lower()

    for word in NOISE_WORDS:
        if word in identity:
            score += 3

    if tag.name in {"nav", "footer", "aside"}:
        score += 2

    role = str(tag.get("role", "")).lower()
    if role in {"navigation", "banner", "contentinfo", "complementary"}:
        score += 2

    return score


def remove_obvious_noise(soup: BeautifulSoup) -> None:
    """
    Remove obvious noise safely.

    IMPORTANT:
    Never decompose elements while iterating over all descendants.
    Decomposing a parent can invalidate its child Tags and later cause:

        'NoneType' object has no attribute 'get'

    Therefore we first collect the elements to remove and only then
    decompose them.
    """

    # Remove comments first.
    for comment in list(
        soup.find_all(string=lambda x: isinstance(x, Comment))
    ):
        comment.extract()

    candidates = []

    # First pass: ONLY inspect elements.
    for tag in list(soup.find_all()):
        if not getattr(tag, "name", None):
            continue

        if tag.name in REMOVE_TAGS:
            candidates.append(tag)
            continue

        # Score only while the Tag is still valid.
        if element_noise_score(tag) >= 4:
            candidates.append(tag)

    # Remove only top-level candidates.
    # If a parent is already scheduled for removal, its children
    # do not need to be decomposed separately.
    candidate_ids = {id(tag) for tag in candidates}

    for tag in candidates:
        parent = getattr(tag, "parent", None)

        skip = False
        while parent is not None:
            if id(parent) in candidate_ids:
                skip = True
                break
            parent = getattr(parent, "parent", None)

        if not skip and getattr(tag, "parent", None) is not None:
            tag.decompose()


def get_title(soup: BeautifulSoup) -> str:
    if soup.title:
        title = clean_space(soup.title.get_text(" ", strip=True))
        if title:
            return title

    h1 = soup.find("h1")
    return clean_space(h1.get_text(" ", strip=True)) if h1 else ""


def text_from_tag(tag) -> str:
    return clean_space(tag.get_text(" ", strip=True))


def is_ui_text(text: str) -> bool:
    normalized = normalize_for_comparison(text)
    if not normalized:
        return True

    if normalized in NOISE_EXACT:
        return True

    return False


def extract_links(container, base_url: str) -> list[dict]:
    links = []
    seen = set()

    for a in container.find_all("a", href=True):
        href = clean_space(a.get("href", ""))
        text = text_from_tag(a)

        if not href:
            continue

        # Keep only links with useful anchor text or clearly useful URLs.
        if not text or is_ui_text(text):
            continue

        key = (href, normalize_for_comparison(text))
        if key in seen:
            continue
        seen.add(key)

        links.append(
            {
                "text": text,
                "url": href if href.startswith(("http://", "https://")) else href,
            }
        )

    return links


def extract_table(table) -> list[list[str]]:
    rows = []

    for tr in table.find_all("tr"):
        cells = []
        for cell in tr.find_all(["th", "td"]):
            value = text_from_tag(cell)
            cells.append(value)

        if any(cells):
            rows.append(cells)

    # A one-cell decorative table is generally not useful.
    if len(rows) == 1 and len(rows[0]) <= 1:
        return []

    return rows


def dedupe_consecutive(items: list[str]) -> list[str]:
    result = []
    previous = None

    for item in items:
        normalized = normalize_for_comparison(item)
        if not normalized:
            continue

        if normalized == previous:
            continue

        result.append(item)
        previous = normalized

    return result


def choose_content_root(soup: BeautifulSoup):
    """Prefer semantic page content over global navigation/footer."""
    main = soup.find("main")
    if main is not None:
        return main

    article = soup.find("article")
    if article is not None:
        return article

    if soup.body is not None:
        return soup.body

    return soup


def collect_raw_blocks(soup: BeautifulSoup) -> list[dict]:
    """
    Extract content blocks in document order.

    We keep the relationship between headings and following content.
    That relationship is important for later RAG retrieval.
    """
    blocks = []

    root = choose_content_root(soup)

    for tag in root.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6",
         "p", "li", "blockquote", "table"],
        recursive=True,
    ):
        # Ignore blocks nested inside another block we already handle.
        if tag.name == "li" and tag.find_parent("li"):
            continue

        if tag.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            text = text_from_tag(tag)
            if text and not is_ui_text(text):
                blocks.append(
                    {
                        "type": "heading",
                        "level": int(tag.name[1]),
                        "text": text,
                    }
                )

        elif tag.name in {"p", "blockquote"}:
            text = text_from_tag(tag)
            if text and not is_ui_text(text):
                blocks.append(
                    {
                        "type": "paragraph",
                        "text": text,
                    }
                )

        elif tag.name == "li":
            text = text_from_tag(tag)
            if text and not is_ui_text(text):
                blocks.append(
                    {
                        "type": "list_item",
                        "text": text,
                    }
                )

        elif tag.name == "table":
            rows = extract_table(tag)
            if rows:
                blocks.append(
                    {
                        "type": "table",
                        "rows": rows,
                    }
                )

    return blocks


def build_sections(blocks: list[dict]) -> list[dict]:
    """
    Convert the document stream into heading-aware sections.

    A section starts at a heading and contains the following content
    until the next heading at the same or higher level.
    """
    sections = []
    current = None
    heading_stack: list[str] = []

    def start_section(heading: str, level: int):
        nonlocal current

        # Remove deeper heading context.
        while len(heading_stack) >= level:
            heading_stack.pop()

        heading_stack.append(heading)

        current = {
            "heading": heading,
            "level": level,
            "section_path": list(heading_stack),
            "paragraphs": [],
            "list_items": [],
            "tables": [],
        }
        sections.append(current)

    for block in blocks:
        if block["type"] == "heading":
            level = block["level"]

            if current is None:
                start_section(block["text"], level)
                continue

            # Every heading becomes a meaningful structural boundary.
            start_section(block["text"], level)

        elif block["type"] == "paragraph":
            if current is None:
                start_section("Introduction", 1)
            current["paragraphs"].append(block["text"])

        elif block["type"] == "list_item":
            if current is None:
                start_section("Introduction", 1)
            current["list_items"].append(block["text"])

        elif block["type"] == "table":
            if current is None:
                start_section("Introduction", 1)
            current["tables"].append(block["rows"])

    # Remove empty sections.
    cleaned = []
    for section in sections:
        if (
            section["paragraphs"]
            or section["list_items"]
            or section["tables"]
        ):
            cleaned.append(section)

    return cleaned


def section_text(section: dict) -> str:
    parts = []

    if section["heading"]:
        parts.append(section["heading"])

    parts.extend(section["paragraphs"])
    parts.extend(section["list_items"])

    for table in section["tables"]:
        for row in table:
            parts.append(" | ".join(row))

    return clean_space(" ".join(parts))


def make_rag_text(title: str, url: str, section: dict) -> str:
    """
    Create a retrieval-friendly representation WITHOUT summarizing.
    It preserves the source wording and section context.
    """
    lines = []

    if title:
        lines.append(f"Page: {title}")

    if section["section_path"]:
        lines.append("Section: " + " > ".join(section["section_path"]))

    if section["paragraphs"]:
        lines.append("")
        lines.extend(section["paragraphs"])

    if section["list_items"]:
        lines.append("")
        lines.extend(f"- {item}" for item in section["list_items"])

    for table in section["tables"]:
        lines.append("")
        for row in table:
            lines.append(" | ".join(row))

    return "\n".join(lines).strip()


def load_metadata(path: Path) -> dict:
    """Load metadata safely; malformed/null JSON must never crash cleaning."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def choose_html(page_dir: Path) -> Path | None:
    rendered = page_dir / "rendered.html"
    raw = page_dir / "raw.html"

    if rendered.exists() and rendered.stat().st_size > 0:
        return rendered

    if raw.exists() and raw.stat().st_size > 0:
        return raw

    return None


def extract_page(page_dir: Path) -> dict | None:
    metadata = load_metadata(page_dir / "metadata.json")
    html_file = choose_html(page_dir)

    if html_file is None:
        return None

    html = html_file.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "lxml")

    remove_obvious_noise(soup)

    title = get_title(soup)
    url = metadata.get("url") or metadata.get("final_url") or ""

    blocks = collect_raw_blocks(soup)
    sections = build_sections(blocks)

    # Remove duplicate paragraphs/list items within each section.
    for section in sections:
        section["paragraphs"] = dedupe_consecutive(section["paragraphs"])
        section["list_items"] = dedupe_consecutive(section["list_items"])

    # Keep sections only if they contain enough meaningful information.
    meaningful_sections = []
    for section in sections:
        text = section_text(section)
        if len(text) >= MIN_SECTION_TEXT:
            meaningful_sections.append(section)

    if not meaningful_sections:
        return {
            "accepted": False,
            "reason": "no_meaningful_content",
            "url": url,
            "title": title,
        }

    full_text = clean_space(" ".join(section_text(s) for s in meaningful_sections))

    if len(full_text) < MIN_PAGE_TEXT:
        return {
            "accepted": False,
            "reason": "page_too_short_after_cleaning",
            "url": url,
            "title": title,
            "text_length": len(full_text),
        }

    # Useful links are extracted after cleaning, from the remaining DOM.
    links = extract_links(choose_content_root(soup), url)

    rag_sections = []
    for index, section in enumerate(meaningful_sections):
        rag_text = make_rag_text(title, url, section)

        rag_sections.append(
            {
                "section_id": index,
                "heading": section["heading"],
                "level": section["level"],
                "section_path": section["section_path"],
                "paragraphs": section["paragraphs"],
                "list_items": section["list_items"],
                "tables": section["tables"],
                "rag_text": rag_text,
                "text_length": len(rag_text),
            }
        )

    return {
        "accepted": True,
        "cleaning_version": CLEANING_VERSION,
        "page_id": page_dir.name,
        "url": url,
        "final_url": metadata.get("final_url"),
        "title": title,
        "sections": rag_sections,
        "links": links,
        "full_text": full_text,
        "quality": {
            "text_length": len(full_text),
            "section_count": len(rag_sections),
            "link_count": len(links),
            "word_count": len(full_text.split()),
        },
        "source": {
            "html_file": str(html_file),
            "source_sha256": sha256_text(html),
            "status_code": metadata.get("status_code"),
        },
    }


def build_repeated_text_index(results: list[dict]) -> Counter:
    """
    Find repeated text across pages.

    This is used for REPORTING and conservative duplicate detection.
    We do not delete a sentence merely because it appears twice.
    """
    counter = Counter()

    for result in results:
        if not result.get("accepted"):
            continue

        seen_on_page = set()

        for section in result.get("sections", []):
            for text in section.get("paragraphs", []):
                normalized = normalize_for_comparison(text)

                if (
                    REPEATED_BOILERPLATE_MIN_LENGTH
                    <= len(normalized)
                    <= REPEATED_BOILERPLATE_MAX_LENGTH
                ):
                    if normalized not in seen_on_page:
                        counter[normalized] += 1
                        seen_on_page.add(normalized)

    return counter


def main() -> None:
    if not INPUT_DIR.exists():
        raise SystemExit(
            f"Input directory not found: {INPUT_DIR.resolve()}\n"
            "Run crawl.py first so swarnasky_raw exists."
        )

    pages_dir = INPUT_DIR / "pages"
    if not pages_dir.exists():
        raise SystemExit(
            f"Pages directory not found: {pages_dir.resolve()}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Start a fresh run so old rejected/error records are not mixed with
    # the current crawl. Existing source data in swarnasky_raw is untouched.
    for stale in (
        OUTPUT_DIR / "rejected_pages.jsonl",
        OUTPUT_DIR / "clean_manifest.jsonl",
        OUTPUT_DIR / "cleaning_report.json",
    ):
        if stale.exists():
            stale.unlink()

    accepted = 0
    rejected = 0
    results = []

    for page_dir in sorted(pages_dir.iterdir()):
        if not page_dir.is_dir():
            continue

        try:
            result = extract_page(page_dir)

            if result is None:
                continue

            results.append(result)

            if not result.get("accepted"):
                rejected += 1
                with (OUTPUT_DIR / "rejected_pages.jsonl").open(
                    "a", encoding="utf-8"
                ) as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

                print(
                    f"REJECTED  {page_dir.name}  "
                    f"{result.get('reason', 'unknown')}"
                )
                continue

            accepted += 1

            out_dir = OUTPUT_DIR / "pages" / page_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)

            out_file = out_dir / "content.json"
            out_file.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            print(
                f"ACCEPTED  {page_dir.name}  "
                f"{result['title']!r}  "
                f"{result['quality']['word_count']} words"
            )

        except Exception as exc:
            rejected += 1
            failure = {
                "accepted": False,
                "reason": f"processing_error:{type(exc).__name__}",
                "page_id": page_dir.name,
                "error": str(exc),
            }

            with (OUTPUT_DIR / "rejected_pages.jsonl").open(
                "a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(failure, ensure_ascii=False) + "\n")

            print(f"ERROR     {page_dir.name}: {exc}")

    repeated = build_repeated_text_index(results)

    repeated_boilerplate = [
        {
            "text": text,
            "page_count": count,
        }
        for text, count in repeated.items()
        if count >= REPEATED_BOILERPLATE_MIN_PAGES
    ]

    manifest_path = OUTPUT_DIR / "clean_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for result in results:
            if not result.get("accepted"):
                continue

            f.write(
                json.dumps(
                    {
                        "page_id": result["page_id"],
                        "url": result["url"],
                        "title": result["title"],
                        "text_length": result["quality"]["text_length"],
                        "word_count": result["quality"]["word_count"],
                        "section_count": result["quality"]["section_count"],
                        "content_file": str(
                            OUTPUT_DIR
                            / "pages"
                            / result["page_id"]
                            / "content.json"
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    report = {
        "cleaning_version": CLEANING_VERSION,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(INPUT_DIR.resolve()),
        "output_directory": str(OUTPUT_DIR.resolve()),
        "pages_accepted": accepted,
        "pages_rejected": rejected,
        "repeated_text_candidates": repeated_boilerplate,
        "rules": {
            "remove_tags": sorted(REMOVE_TAGS),
            "minimum_page_text": MIN_PAGE_TEXT,
            "minimum_section_text": MIN_SECTION_TEXT,
            "duplicate_strategy": (
                "conservative: duplicate detection is reported; "
                "source content is not silently deleted across pages"
            ),
            "rewriting": "none",
            "summarization": "none",
        },
    }

    (OUTPUT_DIR / "cleaning_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n========================================")
    print("CLEANING COMPLETE")
    print("========================================")
    print(f"Accepted pages : {accepted}")
    print(f"Rejected pages : {rejected}")
    print(f"Output         : {OUTPUT_DIR.resolve()}")
    print(f"Report         : {(OUTPUT_DIR / 'cleaning_report.json').resolve()}")
    print()
    print("Next step: inspect the clean output before chunking/embeddings/Qdrant.")


if __name__ == "__main__":
    main()
