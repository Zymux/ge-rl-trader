from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT_DIR / "data" / "news" / "raw"
DERIVED_ROOT = ROOT_DIR / "data" / "news" / "derived"
MAPPINGS_ROOT = ROOT_DIR / "data" / "mappings"

STYLE_TO_ITEMS_PATH = MAPPINGS_ROOT / "style_to_items.json"
ITEM_NAME_TO_ID_PATH = MAPPINGS_ROOT / "item_name_to_id.json"


def _now_utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    if not path.exists():
        return docs
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return docs


def _ensure_dirs() -> None:
    DERIVED_ROOT.mkdir(parents=True, exist_ok=True)
    MAPPINGS_ROOT.mkdir(parents=True, exist_ok=True)


def _load_style_to_items() -> Dict[str, List[int]]:
    if not STYLE_TO_ITEMS_PATH.exists():
        return {}
    with STYLE_TO_ITEMS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, List[int]] = {}
    for k, v in data.items():
        try:
            out[k] = [int(x) for x in v]
        except Exception:
            out[k] = []
    return out


def _load_item_name_to_id() -> Dict[str, int]:
    if not ITEM_NAME_TO_ID_PATH.exists():
        return {}
    with ITEM_NAME_TO_ID_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, int] = {}
    for name, raw_id in data.items():
        try:
            out[name.lower()] = int(raw_id)
        except Exception:
            continue
    return out


# ---------- Text utilities ----------


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _doc_text(doc: Dict[str, Any]) -> str:
    title = doc.get("title") or ""
    text = doc.get("text") or ""
    joined = f"{title}\n\n{text}".strip()
    return joined


# ---------- Item + style extraction ----------


def extract_styles(text: str) -> Set[str]:
    """
    Very lightweight style / gear-category tags.
    """
    t = text.lower()
    tags: Set[str] = set()

    magic_tokens = ["magic", "spell", "spells", "mage", "mages"]
    ranged_tokens = ["ranged", "range", "bow", "bows", "crossbow", "crossbows", "xbow", "xbows", "bolts", "arrows"]
    melee_tokens = ["melee"]
    stab_tokens = ["stab", "stabbing"]
    slash_tokens = ["slash", "slashing"]
    crush_tokens = ["crush", "crushing", "blunt"]

    rune_tokens = ["rune", "runes"]
    ammo_tokens = ["ammo", "ammunition"]
    food_tokens = ["food", "shark", "anglerfish", "karambwan"]
    potion_tokens = ["potion", "potions", "brew", "brews", "restore", "restores", "prayer potion", "super restore"]

    for token in magic_tokens:
        if token in t:
            tags.add("magic")
            break
    for token in ranged_tokens:
        if token in t:
            tags.add("ranged")
            break
    for token in melee_tokens:
        if token in t:
            tags.add("melee")
            break
    for token in stab_tokens:
        if token in t:
            tags.add("stab")
            break
    for token in slash_tokens:
        if token in t:
            tags.add("slash")
            break
    for token in crush_tokens:
        if token in t:
            tags.add("crush")
            break

    for token in rune_tokens:
        if token in t:
            tags.add("runes")
            break
    for token in ammo_tokens:
        if token in t:
            tags.add("ammo")
            break
    for token in food_tokens:
        if token in t:
            tags.add("food")
            break
    for token in potion_tokens:
        if token in t:
            tags.add("potions")
            break

    return tags


def extract_item_mentions(text: str, item_name_to_id: Dict[str, int]) -> List[int]:
    """
    Extract item_ids by matching known item names from a small manual mapping.
    """
    if not item_name_to_id:
        return []

    t = text.lower()
    found_ids: Set[int] = set()

    for name, item_id in item_name_to_id.items():
        # Require whole-word-ish matches to reduce noise.
        pattern = r"\b" + re.escape(name) + r"\b"
        if re.search(pattern, t):
            found_ids.add(item_id)

    return sorted(found_ids)


# ---------- Event classification ----------


def classify_events_for_doc(
    doc: Dict[str, Any],
    items_mentioned: Sequence[int],
    styles: Set[str],
) -> List[Dict[str, Any]]:
    """
    Rule-based event classification.
    Returns a list of event dicts for this document.
    """
    text = _normalize_text(_doc_text(doc)).lower()
    events: List[Dict[str, Any]] = []

    source = doc.get("source") or ""
    url = doc.get("url") or ""
    title = (doc.get("title") or "").strip()

    def add_event(
        ev_type: str,
        confidence: float,
        detail_reason: str,
        extra_tags: Optional[Iterable[str]] = None,
    ) -> None:
        tags: List[str] = []
        if extra_tags:
            tags.extend(extra_tags)
        tags.extend(sorted(styles))
        events.append(
            {
                "type": ev_type,
                "confidence": float(confidence),
                "details": detail_reason,
                "tags": sorted(set(tags)),
                "items_mentioned": list(items_mentioned),
                "source": source,
                "source_title": title,
                "source_url": url,
            }
        )

    # NEW_BOSS / NEW_RAID / NEW_CONTENT
    if "new boss" in text or "new raid" in text:
        add_event("NEW_BOSS", 0.8, 'Keyword match: "new boss"/"new raid".', ["pvm", "release"])
    elif "raid" in text and ("release" in text or "coming" in text):
        add_event("NEW_RAID", 0.7, 'Heuristic: raid + release/coming.', ["pvm", "release"])
    elif "new content" in text or "content update" in text:
        add_event("NEW_CONTENT", 0.6, 'Heuristic: "new content"/"content update".', ["release"])

    # WEAKNESS_CHANGE
    if "weak to" in text or "weakness" in text or "more vulnerable to" in text:
        add_event("WEAKNESS_CHANGE", 0.7, "Heuristic: weakness/weak to/vulnerable phrasing.", [])

    # ITEM_BUFF / ITEM_NERF
    buff_tokens = ["buffed", "buff", "increase damage", "increased damage", "stronger", "more effective"]
    nerf_tokens = ["nerfed", "nerf", "reduced damage", "less damage", "weaker", "reduced effectiveness"]

    if any(tok in text for tok in buff_tokens):
        add_event("ITEM_BUFF", 0.7, "Heuristic: buff/increase damage wording.", [])
    if any(tok in text for tok in nerf_tokens):
        add_event("ITEM_NERF", 0.7, "Heuristic: nerf/reduced damage wording.", [])

    # DROP_TABLE_CHANGE
    if "drop table" in text or "droptable" in text or "now drops" in text or "removed from the drop" in text:
        add_event("DROP_TABLE_CHANGE", 0.7, "Heuristic: drop table / now drops / removed from drop.", [])

    # SUPPLY_SINK
    supply_tokens = [
        "now requires",
        "consumed on use",
        "consumes on use",
        "consumed when",
        "degrade",
        "degrades",
        "degradation",
        "charges are consumed",
        "charge is consumed",
        "item sink",
    ]
    if any(tok in text for tok in supply_tokens):
        add_event("SUPPLY_SINK", 0.6, "Heuristic: degradation/consumed/charge wording.", [])

    # BUG / HOTFIX
    if "hotfix" in text or "hot-fix" in text or "bug fix" in text or "bugfix" in text:
        add_event("BUG/HOTFIX", 0.9, "Heuristic: hotfix / bug fix wording.", ["uncertainty"])
    elif ("fixed an issue" in text or "fixed a bug" in text or "fixing an issue" in text) and "BUG/HOTFIX" not in [
        e["type"] for e in events
    ]:
        add_event("BUG/HOTFIX", 0.7, 'Heuristic: "fixed an issue/bug".', ["uncertainty"])

    return events


# ---------- Watchlist + risk flags ----------


def build_watchlist(
    events: Sequence[Dict[str, Any]],
    style_to_items: Dict[str, List[int]],
) -> List[Dict[str, Any]]:
    """
    Turn events into a candidate watchlist with simple direction heuristics.
    """
    watch: Dict[int, Dict[str, Any]] = {}

    def add_item(item_id: int, reason: str, impact: str) -> None:
        if item_id in watch:
            # Concatenate reasons; keep strongest impact if different.
            prev = watch[item_id]
            if reason not in prev["reason"]:
                prev["reason"] += "; " + reason
            # Prefer demand_up/supply_up over demand_down when mixing.
            if prev["expected_impact"] != impact and prev["expected_impact"] == "demand_down":
                prev["expected_impact"] = impact
        else:
            watch[item_id] = {
                "item_id": int(item_id),
                "reason": reason,
                "expected_impact": impact,
            }

    for ev in events:
        ev_type = ev.get("type")
        tags = set(ev.get("tags") or [])
        items = ev.get("items_mentioned") or []

        # Direct item effects
        if ev_type == "ITEM_BUFF":
            for iid in items:
                add_item(int(iid), "Mentioned in ITEM_BUFF event", "demand_up")
        elif ev_type == "ITEM_NERF":
            for iid in items:
                add_item(int(iid), "Mentioned in ITEM_NERF event", "demand_down")
        elif ev_type == "DROP_TABLE_CHANGE":
            for iid in items:
                add_item(int(iid), "Mentioned in DROP_TABLE_CHANGE event", "supply_up")

        # Style-based inferred demand for supplies / gear
        style_hits: Set[str] = set()
        for style in ["magic", "ranged", "melee", "stab", "slash", "crush", "runes", "ammo"]:
            if style in tags:
                style_hits.add(style)

        if ev_type in {"NEW_BOSS", "NEW_RAID", "NEW_CONTENT", "WEAKNESS_CHANGE"} and style_hits:
            for style in style_hits:
                style_key = style
                if style_key in {"runes"}:
                    style_key = "magic"
                if style_key in {"ammo"}:
                    style_key = "ranged"

                candidates = style_to_items.get(style_key, [])
                for iid in candidates:
                    add_item(
                        int(iid),
                        f"Style {style_key} implied by {ev_type} event",
                        "demand_up",
                    )

    return sorted(watch.values(), key=lambda x: x["item_id"])


def infer_risk_flags(
    events: Sequence[Dict[str, Any]],
    docs: Sequence[Dict[str, Any]],
) -> List[str]:
    flags: Set[str] = set()

    if any(ev.get("type") == "BUG/HOTFIX" for ev in events):
        flags.add("hotfix_rolling")

    bug_words = ["bug", "broken", "crash", "crashing"]
    bug_posts = 0
    for doc in docs:
        text = _doc_text(doc).lower()
        if any(w in text for w in bug_words):
            bug_posts += 1
    if bug_posts >= 3:
        flags.add("bug_reports")

    return sorted(flags)


# ---------- Main pipeline ----------


def build_context_card_for_date(target_date: dt.date) -> Dict[str, Any]:
    _ensure_dirs()

    date_str = target_date.isoformat()
    raw_dir = RAW_ROOT / date_str

    official_path = raw_dir / "official_osrs.jsonl"
    reddit_path = raw_dir / "reddit_2007scape.jsonl"

    official_docs = _load_jsonl(official_path)
    reddit_docs = _load_jsonl(reddit_path)
    all_docs = official_docs + reddit_docs

    style_to_items = _load_style_to_items()
    item_name_to_id = _load_item_name_to_id()

    events: List[Dict[str, Any]] = []

    for doc in all_docs:
        full_text = _doc_text(doc)
        styles = extract_styles(full_text)
        items_mentioned = extract_item_mentions(full_text, item_name_to_id)
        doc_events = classify_events_for_doc(doc, items_mentioned, styles)
        events.extend(doc_events)

    watchlist = build_watchlist(events, style_to_items)
    risk_flags = infer_risk_flags(events, reddit_docs)

    context = {
        "as_of": _now_utc_iso(),
        "date": date_str,
        "sources_seen": {
            "official": len(official_docs),
            "reddit": len(reddit_docs),
        },
        "events": events,
        "watchlist": watchlist,
        "risk_flags": risk_flags,
    }

    out_path = DERIVED_ROOT / f"context_card_{date_str}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(context, f, ensure_ascii=False, indent=2)

    return context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Turn raw news JSONL into a structured context card (events + watchlist + risk flags)."
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target calendar date (YYYY-MM-DD). Reads from data/news/raw/<date>/ and writes "
        "data/news/derived/context_card_<date>.json. Defaults to today (UTC).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.date:
        target_date = dt.date.fromisoformat(args.date)
    else:
        target_date = dt.datetime.now(dt.timezone.utc).date()

    context = build_context_card_for_date(target_date)
    out_path = DERIVED_ROOT / f"context_card_{target_date.isoformat()}.json"
    print(f"Wrote context card with {len(context['events'])} events and {len(context['watchlist'])} watchlist items:")
    print(f"  {out_path}")


if __name__ == "__main__":
    main()

