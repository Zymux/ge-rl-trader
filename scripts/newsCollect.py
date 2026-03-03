from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional

import requests
from bs4 import BeautifulSoup


ROOT_DIR = Path(__file__).resolve().parents[1]
NEWS_ROOT = ROOT_DIR / "data" / "news" / "raw"


@dataclass
class NewsDoc:
    source: str
    fetched_utc: str
    title: str
    text: str
    url: str
    published_utc: Optional[str] = None
    score: Optional[int] = None
    num_comments: Optional[int] = None


def _now_utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _ensure_out_dir(target_date: dt.date) -> Path:
    out_dir = NEWS_ROOT / target_date.isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _write_jsonl(path: Path, docs: Iterable[NewsDoc]) -> None:
    if not docs:
        return
    with path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")


# ---------- Official OSRS news (A: Official / authoritative) ----------

OSRS_NEWS_ARCHIVE = "https://secure.runescape.com/m=news/archive"


def fetch_official_archive_html(target_date: dt.date) -> str:
    params = {"oldschool": "1", "month": str(target_date.month), "year": str(target_date.year)}
    resp = requests.get(OSRS_NEWS_ARCHIVE, params=params, timeout=20)
    resp.raise_for_status()
    return resp.text


def extract_official_links(archive_html: str, limit: int) -> List[str]:
    soup = BeautifulSoup(archive_html, "html.parser")
    links: List[str] = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Only keep Old School news posts
        if "/m=news/" in href and "oldschool=1" in href:
            if href.startswith("http"):
                url = href
            else:
                url = "https://secure.runescape.com" + href
            if url not in links:
                links.append(url)
        if len(links) >= limit:
            break

    return links


DATE_PATTERN = re.compile(r"\b(\d{1,2}\s+[A-Z][a-z]+\s+\d{4})\b")


def _extract_date_from_text(text: str) -> Optional[str]:
    match = DATE_PATTERN.search(text)
    if not match:
        return None
    raw = match.group(1)
    try:
        parsed = dt.datetime.strptime(raw, "%d %B %Y").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None
    return parsed.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def fetch_official_post(url: str, fetched_iso: str) -> NewsDoc:
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    title_tag = soup.find("h1") or soup.find("h2")
    title = title_tag.get_text(strip=True) if title_tag else url

    body_text = soup.get_text(separator="\n")
    body_text = re.sub(r"\n{2,}", "\n\n", body_text).strip()

    published_iso = _extract_date_from_text(body_text)

    return NewsDoc(
        source="official:osrs_news",
        fetched_utc=fetched_iso,
        published_utc=published_iso,
        title=title,
        text=body_text,
        url=url,
        score=None,
        num_comments=None,
    )


def collect_official_news(target_date: dt.date, max_posts: int) -> List[NewsDoc]:
    fetched_iso = _now_utc_iso()
    html = fetch_official_archive_html(target_date)
    links = extract_official_links(html, limit=max_posts)
    docs: List[NewsDoc] = []
    for url in links:
        try:
            docs.append(fetch_official_post(url, fetched_iso=fetched_iso))
        except Exception as exc:
            # Best-effort: log to console and continue
            print(f"[official] Failed to fetch {url}: {exc}")
    return docs


# ---------- Reddit community signal (B: Community signal) ----------

REDDIT_BASE = "https://www.reddit.com"
SUBREDDIT_PATH = "/r/2007scape/new.json"

REDDIT_FLAIR_WHITELIST = {
    "Update",
    "News",
    "PSA",
    "Bug",
    "Discussion",
}


def collect_reddit_posts(max_posts: int) -> List[NewsDoc]:
    fetched_iso = _now_utc_iso()
    url = REDDIT_BASE + SUBREDDIT_PATH
    headers = {
        "User-Agent": "ge-rl-trader-news-collector/0.1 (contact: local-script)",
    }
    params = {"limit": str(max_posts)}

    resp = requests.get(url, headers=headers, params=params, timeout=20)
    resp.raise_for_status()
    payload = resp.json()

    children = payload.get("data", {}).get("children", [])
    docs: List[NewsDoc] = []

    for child in children:
        data = child.get("data", {})
        flair = (data.get("link_flair_text") or "").strip()
        if flair and flair not in REDDIT_FLAIR_WHITELIST:
            continue

        title = data.get("title") or ""
        selftext = data.get("selftext") or ""
        text = (title + "\n\n" + selftext).strip()
        if not text:
            continue

        created_utc = data.get("created_utc")
        if isinstance(created_utc, (int, float)):
            published = dt.datetime.fromtimestamp(created_utc, tz=dt.timezone.utc)
            published_iso = published.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        else:
            published_iso = None

        permalink = data.get("permalink") or ""
        if permalink.startswith("http"):
            url_full = permalink
        else:
            url_full = REDDIT_BASE + permalink

        docs.append(
            NewsDoc(
                source="reddit:r/2007scape",
                fetched_utc=fetched_iso,
                published_utc=published_iso,
                title=title,
                text=text,
                url=url_full,
                score=int(data.get("score") or 0),
                num_comments=int(data.get("num_comments") or 0),
            )
        )

    return docs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect OSRS news + community posts into JSONL for later event extraction."
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target calendar date (YYYY-MM-DD) for the run. "
        "Controls output folder name and OSRS archive month/year. Defaults to today (UTC).",
    )
    parser.add_argument(
        "--max-official",
        type=int,
        default=10,
        help="Maximum number of official OSRS news posts to fetch from the archive month.",
    )
    parser.add_argument(
        "--max-reddit",
        type=int,
        default=50,
        help="Maximum number of Reddit posts to fetch from r/2007scape (filtered by flair).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.date:
        target_date = dt.date.fromisoformat(args.date)
    else:
        target_date = dt.datetime.now(dt.timezone.utc).date()

    out_dir = _ensure_out_dir(target_date)
    print(f"Writing raw news docs to: {out_dir}")

    official_docs = collect_official_news(target_date=target_date, max_posts=args.max_official)
    reddit_docs = collect_reddit_posts(max_posts=args.max_reddit)

    _write_jsonl(out_dir / "official_osrs.jsonl", official_docs)
    _write_jsonl(out_dir / "reddit_2007scape.jsonl", reddit_docs)

    print(f"Collected {len(official_docs)} official OSRS posts.")
    print(f"Collected {len(reddit_docs)} reddit posts (filtered by flair).")
    print("Done.")


if __name__ == "__main__":
    main()

