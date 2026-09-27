"""
Swarnasky Step 1 — complete website discovery + raw archival.

IMPORTANT:
- This script does NOT clean, summarize, chunk, embed, or send data to Qdrant.
- It preserves the original HTTP response HTML when available.
- It also saves a browser-rendered HTML copy for JavaScript-generated content.
- It saves URL/link metadata so the crawl can be audited later.
- It respects robots.txt by default.

Run:
    pip install -r requirements.txt
    playwright install chromium
    python crawl_swarnasky.py

Output:
    swarnasky_raw/
      pages/<sha256-prefix>/
        raw.html
        rendered.html
        metadata.json
      crawl_manifest.jsonl
      failed_urls.jsonl
      discovered_urls.txt
      robots.txt
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import (
    parse_qsl,
    urlencode,
    urldefrag,
    urljoin,
    urlparse,
    urlunparse,
)
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright


START_URL = "https://www.swarnasky.com/"
OUTPUT_DIR = Path("swarnasky_raw")

# Safety/coverage controls. Increase MAX_PAGES if the site is larger.
MAX_PAGES = 5000
REQUEST_TIMEOUT = 30
DELAY_SECONDS = 0.5
USER_AGENT = "SwarnaskyKnowledgeCrawler/1.0 (+respectful-site-archive)"
RESPECT_ROBOTS = True

# Keep normal page URLs. These are not knowledge pages.
SKIP_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
    ".mp4", ".webm", ".mov", ".avi", ".mp3", ".wav",
    ".zip", ".rar", ".7z", ".tar", ".gz",
    ".css", ".js", ".map",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
}

# Query parameters that are commonly tracking/noise.
DROP_QUERY_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "msclkid", "mc_cid", "mc_eid",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_url(url: str) -> str | None:
    if not url:
        return None

    url, _fragment = urldefrag(url)
    p = urlparse(url)

    if p.scheme not in {"http", "https"}:
        return None

    host = (p.hostname or "").lower()
    if host not in {"swarnasky.com", "www.swarnasky.com"}:
        return None

    # Canonicalize www/non-www to the starting domain.
    host = "www.swarnasky.com"

    path = p.path or "/"
    path = re.sub(r"/{2,}", "/", path)
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    query_pairs = []
    for k, v in parse_qsl(p.query, keep_blank_values=True):
        if k.lower() not in DROP_QUERY_PARAMS:
            query_pairs.append((k, v))

    query = urlencode(sorted(query_pairs))
    return urlunparse(("https", host, path, "", query, ""))


def is_html_candidate(url: str) -> bool:
    path = urlparse(url).path.lower()
    return not any(path.endswith(ext) for ext in SKIP_EXTENSIONS)


def page_dir(url: str) -> Path:
    return OUTPUT_DIR / "pages" / sha256_text(url)[:16]


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append_jsonl(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    found = []

    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        if not href:
            continue

        absolute = normalize_url(urljoin(base_url, href))
        if absolute and is_html_candidate(absolute):
            found.append(absolute)

    return sorted(set(found))


def discover_sitemaps(session: requests.Session) -> list[str]:
    candidates = [
        "https://www.swarnasky.com/sitemap.xml",
        "https://www.swarnasky.com/sitemap_index.xml",
        "https://www.swarnasky.com/wp-sitemap.xml",
    ]

    results = []

    for url in candidates:
        try:
            r = session.get(url, timeout=REQUEST_TIMEOUT)
            if r.ok and ("xml" in r.headers.get("content-type", "").lower()
                         or r.text.lstrip().startswith("<?xml")
                         or "<urlset" in r.text[:1000]
                         or "<sitemapindex" in r.text[:1000]):
                results.append(url)
        except requests.RequestException:
            pass

    return results


def parse_sitemap(url: str, session: requests.Session, seen=None) -> set[str]:
    """Recursively parse sitemap and sitemap indexes."""
    if seen is None:
        seen = set()

    if url in seen:
        return set()
    seen.add(url)

    try:
        r = session.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
    except requests.RequestException:
        return set()

    soup = BeautifulSoup(r.content, "xml")
    urls = set()

    if soup.find("sitemapindex"):
        for loc in soup.find_all("loc"):
            child = normalize_url(loc.get_text(strip=True))
            if child:
                urls |= parse_sitemap(child, session, seen)
    else:
        for loc in soup.find_all("loc"):
            page = normalize_url(loc.get_text(strip=True))
            if page and is_html_candidate(page):
                urls.add(page)

    return urls


def load_robots(session: requests.Session) -> RobotFileParser:
    robots_url = "https://www.swarnasky.com/robots.txt"
    rp = RobotFileParser()

    try:
        r = session.get(robots_url, timeout=REQUEST_TIMEOUT)
        (OUTPUT_DIR / "robots.txt").write_bytes(r.content)

        if r.ok:
            rp.parse(r.text.splitlines())
        else:
            # If robots cannot be fetched, default to allowing crawl but record it.
            rp.parse([])
    except requests.RequestException:
        rp.parse([])

    return rp


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    robots = load_robots(session)

    # Seed with homepage + sitemap URLs.
    seeds = {normalize_url(START_URL)}
    sitemap_urls = discover_sitemaps(session)

    for sitemap in sitemap_urls:
        seeds |= parse_sitemap(sitemap, session)

    seeds = {u for u in seeds if u}

    queue = deque(sorted(seeds))
    queued = set(queue)
    visited: set[str] = set()

    (OUTPUT_DIR / "crawl_started.json").write_text(
        json.dumps(
            {
                "started_at": now_iso(),
                "start_url": START_URL,
                "max_pages": MAX_PAGES,
                "robots_respected": RESPECT_ROBOTS,
                "sitemaps_found": sitemap_urls,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Seeds: {len(queue)}")
    print(f"Sitemaps: {len(sitemap_urls)}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=USER_AGENT,
            java_script_enabled=True,
            ignore_https_errors=False,
        )

        while queue and len(visited) < MAX_PAGES:
            url = queue.popleft()
            queued.discard(url)

            if url in visited:
                continue

            if RESPECT_ROBOTS and not robots.can_fetch(USER_AGENT, url):
                append_jsonl(
                    OUTPUT_DIR / "failed_urls.jsonl",
                    {
                        "url": url,
                        "reason": "blocked_by_robots_txt",
                        "timestamp": now_iso(),
                    },
                )
                visited.add(url)
                continue

            visited.add(url)

            print(f"[{len(visited)}] {url}")

            meta = {
                "url": url,
                "crawl_started_at": now_iso(),
                "status_code": None,
                "content_type": None,
                "final_url": None,
                "raw_bytes": 0,
                "rendered_bytes": 0,
                "raw_sha256": None,
                "rendered_sha256": None,
                "discovered_links": [],
                "error": None,
            }

            raw_html = None

            # 1) Preserve original HTTP response.
            try:
                r = session.get(
                    url,
                    timeout=REQUEST_TIMEOUT,
                    allow_redirects=True,
                )
                meta["status_code"] = r.status_code
                meta["content_type"] = r.headers.get("content-type")
                meta["final_url"] = normalize_url(r.url) or r.url

                if r.ok and "text/html" in r.headers.get("content-type", "").lower():
                    raw_html = r.text
                    meta["raw_bytes"] = len(r.content)
                    meta["raw_sha256"] = hashlib.sha256(r.content).hexdigest()
            except requests.RequestException as e:
                meta["error"] = f"requests: {type(e).__name__}: {e}"

            # 2) Render with Chromium so JS-generated content is also captured.
            rendered_html = None
            try:
                page = context.new_page()
                response = page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=REQUEST_TIMEOUT * 1000,
                )
                try:
                    page.wait_for_load_state("networkidle", timeout=10000)
                except Exception:
                    pass

                rendered_html = page.content()
                meta["rendered_bytes"] = len(rendered_html.encode("utf-8"))
                meta["rendered_sha256"] = sha256_text(rendered_html)

                browser_url = normalize_url(page.url)
                if browser_url:
                    meta["browser_final_url"] = browser_url

                page.close()
            except Exception as e:
                if meta["error"]:
                    meta["error"] += f" | playwright: {type(e).__name__}: {e}"
                else:
                    meta["error"] = f"playwright: {type(e).__name__}: {e}"

            # Nothing useful was retrieved.
            if raw_html is None and rendered_html is None:
                append_jsonl(OUTPUT_DIR / "failed_urls.jsonl", meta)
                time.sleep(DELAY_SECONDS)
                continue

            # Prefer raw HTML for the archival file; rendered HTML is separately preserved.
            out = page_dir(url)
            out.mkdir(parents=True, exist_ok=True)

            if raw_html is not None:
                (out / "raw.html").write_text(raw_html, encoding="utf-8")

            if rendered_html is not None:
                (out / "rendered.html").write_text(
                    rendered_html,
                    encoding="utf-8",
                )

            # Extract links from BOTH versions.
            links = set()
            if raw_html:
                links.update(extract_links(raw_html, url))
            if rendered_html:
                links.update(extract_links(rendered_html, url))

            links = sorted(links)
            meta["discovered_links"] = links

            write_json(out / "metadata.json", meta)
            append_jsonl(OUTPUT_DIR / "crawl_manifest.jsonl", meta)

            with (OUTPUT_DIR / "discovered_urls.txt").open(
                "a", encoding="utf-8"
            ) as f:
                for link in links:
                    f.write(link + "\n")

            # Add newly discovered URLs.
            for link in links:
                if link not in visited and link not in queued:
                    queue.append(link)
                    queued.add(link)

            time.sleep(DELAY_SECONDS)

        browser.close()

    # Final deduplicated URL list.
    discovered = set()
    manifest = OUTPUT_DIR / "crawl_manifest.jsonl"
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
                discovered.add(item["url"])
                discovered.update(item.get("discovered_links", []))
            except Exception:
                pass

    (OUTPUT_DIR / "discovered_urls.txt").write_text(
        "\n".join(sorted(discovered)) + ("\n" if discovered else ""),
        encoding="utf-8",
    )

    write_json(
        OUTPUT_DIR / "crawl_finished.json",
        {
            "finished_at": now_iso(),
            "pages_visited": len(visited),
            "unique_urls_discovered": len(discovered),
            "sitemaps_found": sitemap_urls,
        },
    )

    print("\nDONE")
    print(f"Visited: {len(visited)}")
    print(f"Discovered: {len(discovered)}")
    print(f"Output: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
