"""
Build data/mappings/style_to_items.json from the OSRS Wiki Item IDs page.

Fetches the full item list via the wiki's parse API, extracts (name, id) pairs,
categorizes by combat/style keywords, and writes style_to_items.json.

Run from repo root:
  python -m scripts.build_style_to_items
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import requests
from bs4 import BeautifulSoup


ROOT_DIR = Path(__file__).resolve().parents[1]
MAPPINGS_ROOT = ROOT_DIR / "data" / "mappings"
STYLE_TO_ITEMS_PATH = MAPPINGS_ROOT / "style_to_items.json"

WIKI_PARSE_URL = "https://oldschool.runescape.wiki/api.php"


def fetch_item_list() -> List[Tuple[str, int]]:
    """Fetch Item IDs page via parse API and return [(item_name, item_id), ...]."""
    params = {
        "action": "parse",
        "page": "Item_IDs",
        "prop": "text",
        "format": "json",
        "origin": "*",
    }
    headers = {"User-Agent": "ge-rl-trader-wiki-fetch/0.1 (local script)"}
    resp = requests.get(WIKI_PARSE_URL, params=params, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    html = data.get("parse", {}).get("text", {}).get("*", "")
    if not html:
        raise ValueError("No parse text in API response")

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="wikitable")
    if not table:
        raise ValueError("No wikitable found in page")
    tbody = table.find("tbody") or table
    rows = tbody.find_all("tr")
    pairs: List[Tuple[str, int]] = []
    for tr in rows:
        cells = tr.find_all("td")
        if len(cells) != 2:
            continue
        name_cell, id_cell = cells[0], cells[1]
        name = name_cell.get_text(strip=True)
        if not name:
            continue
        id_text = id_cell.get_text(strip=True).replace(",", " ").strip()
        id_match = re.match(r"^(\d+)", id_text)
        if not id_match:
            continue
        item_id = int(id_match.group(1))
        if item_id >= 1_000_000:
            continue
        pairs.append((name, item_id))
    return pairs


def categorize(pairs: List[Tuple[str, int]]) -> Dict[str, List[int]]:
    """Assign each item to one or more style buckets by name keywords."""
    magic_ids: Set[int] = set()
    ranged_ids: Set[int] = set()
    stab_ids: Set[int] = set()
    slash_ids: Set[int] = set()
    crush_ids: Set[int] = set()

    magic_keywords = (
        "staff", "wand", "trident", "sceptre", "tome", "battlestaff", "staff of",
        "mage hat", "mage cape", "robe top", "robe bottom", "robe skirt", "robes",
        "ahrim", "ancestral", "mystic", "infinity", "kodai", "nightmare staff",
        "eldritch", "volatile", "dawnbringer", "sanguinesti", "crystal staff",
        "ibans", "slayer staff", "lunar staff", "ancient staff", "druidic",
        "rune (", " rune)", " runes)", " rune ", "air rune", "water rune", "earth rune",
        "fire rune", "body rune", "mind rune", "death rune", "blood rune", "soul rune",
        "chaos rune", "nature rune", "law rune", "cosmic rune", "astral rune",
        "wrath rune", "mist rune", "dust rune", "mud rune", "smoke rune", "steam rune",
        "lava rune", "armadyl rune", "aether rune"
    )
    ranged_keywords = (
        "bolt", "arrow", "bow", "crossbow", "dart", "javelin", "chinchompa",
        "shortbow", "longbow", "ballista", "blowpipe", "range top", "range legs",
        "range coif", "vambraces", "d'hide", "dragonhide", "chaps", "accumulator",
        "anguish", "pegasian", "armadyl crossbow", "twisted bow", "craw's bow",
        "webweaver", "dragon crossbow", "armadyl", "karil", "crystal bow"
    )
    stab_keywords = (
        "dagger", "spear", "hasta", "rapier", "zamorakian hasta", "dragon hasta",
        "abyssal dagger", "bone dagger", "leaf-bladed", "keris", "stabbing"
    )
    slash_keywords = (
        "scimitar", "whip", "longsword", "2h sword", "blade", "scythe",
        "abyssal whip", "tentacle", "dragon scimitar", "dragon longsword",
        "elder maul", "godsword", "sarachnis", "fang (", "osmumten"
    )
    crush_keywords = (
        "mace", "warhammer", "maul", "bludgeon", "battleaxe", "granite hammer",
        "elder maul", "inquisitor", "dragon mace", "barrelchest", "gadderhammer"
    )

    def norm(s: str) -> str:
        return s.lower().strip()

    for name, item_id in pairs:
        n = norm(name)
        # Avoid armour/weapon that are "rune" (metal) not "rune" (magic)
        if any(k in n for k in ("rune plate", "rune full", "rune chain", "rune kite", "rune legs", "rune skirt", "rune med", "rune scimitar", "rune 2h", "rune sword", "rune dagger", "rune mace", "rune battleaxe", "rune warhammer", "rune pickaxe", "rune axe", "rune nail", "rune arrow", "rune bolt", "rune dart", "rune javelin", "rune thrown", "rune limb")):
            pass  # skip adding to magic for these
        elif any(m in n for m in magic_keywords) or (n.endswith(" rune") and "arrow" not in n and "bolt" not in n):
            magic_ids.add(item_id)
        if any(r in n for r in ranged_keywords):
            ranged_ids.add(item_id)
        if any(s in n for s in stab_keywords):
            stab_ids.add(item_id)
        if any(s in n for s in slash_keywords):
            slash_ids.add(item_id)
        if any(c in n for c in crush_keywords):
            crush_ids.add(item_id)

    melee_ids = stab_ids | slash_ids | crush_ids
    return {
        "magic": sorted(magic_ids),
        "ranged": sorted(ranged_ids),
        "melee": sorted(melee_ids),
        "stab": sorted(stab_ids),
        "slash": sorted(slash_ids),
        "crush": sorted(crush_ids),
    }


def main() -> None:
    MAPPINGS_ROOT.mkdir(parents=True, exist_ok=True)
    print("Fetching Item IDs from OSRS Wiki...")
    pairs = fetch_item_list()
    print(f"Parsed {len(pairs)} (name, id) pairs.")
    result = categorize(pairs)
    for style, ids in result.items():
        print(f"  {style}: {len(ids)} items")
    with STYLE_TO_ITEMS_PATH.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {STYLE_TO_ITEMS_PATH}")


if __name__ == "__main__":
    main()
