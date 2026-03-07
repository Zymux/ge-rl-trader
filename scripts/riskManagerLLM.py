"""
LLM-assisted risk manager: same bounded knobs as riskManager.py, but from an LLM.

Reads context_card_YYYY-MM-DD.json (and optional health), sends a prompt to an LLM,
parses the response as risk_config JSON, then clamps all values to the same bounds
as the rule-based riskManager. Writes risk_config_llm_<date>.json so you can
compare: no manager | rule-based | LLM-assisted.

Requires OPENAI_API_KEY in the environment. Uses OpenAI Chat Completions API
(requests only; no extra dependency).
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
DERIVED_ROOT = ROOT_DIR / "data" / "news" / "derived"

# Same bounds as riskManager.py (hard limits)
MAX_POSITION_GP_RANGE = (250_000.0, 2_000_000.0)
MAX_UNITS_PER_TRADE_RANGE = (1.0, 500.0)
MAX_OPEN_ORDERS_RANGE = (1, 3)
SPREAD_GUARD_PCT_RANGE = (0.0, 0.05)
AGGRESSION_RANGE = (0.0, 1.0)
WATCHLIST_BIAS_RANGE = (-0.25, 0.25)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _parse_json_from_response(text: str) -> Dict[str, Any]:
    """Extract JSON from LLM response, optionally inside markdown code block."""
    text = text.strip()
    # Strip optional markdown code block
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    return json.loads(text)


def _call_llm(system: str, user: str, *, api_key: str, model: str, base_url: Optional[str] = None) -> str:
    """Call OpenAI Chat Completions API; return content of the first message."""
    url = (base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "max_tokens": 2000,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    choice = data.get("choices") or []
    if not choice:
        raise ValueError("No choices in LLM response")
    msg = choice[0].get("message") or {}
    return (msg.get("content") or "").strip()


def _build_prompt(context_card: Dict[str, Any], health: Optional[Dict[str, Any]]) -> tuple[str, str]:
    system = """You are a risk manager for a trading simulator. Given a context card (news events, watchlist, risk flags) and optional health metrics, you must output a single JSON object with exactly these keys. All numeric values will be clamped to safe bounds by the system; you only need to choose sensible values.

Output keys (no extra keys):
- risk_mode: "normal" or "cautious"
- max_position_gp: number (max position size in gp)
- max_units_per_trade: number
- max_open_orders: integer 1-3
- spread_guard_pct: number 0-0.05 (block trades when spread is wider than this)
- aggression: number 0-1 (pricing aggression)
- watchlist_bias: object mapping item_id (as string) to a number in [-0.25, 0.25] (e.g. {"560": 0.1, "565": -0.1}). Only include items from the watchlist.
- focus_items: array of item_id integers (e.g. [560, 565]). Typically items with demand_up.

Rules to follow:
- If risk_flags include "hotfix_rolling" or "bug_reports", set risk_mode to "cautious", lower max_position_gp (e.g. 500000), and use a stricter spread_guard_pct (e.g. 0.04).
- For demand_up watchlist items, use positive watchlist_bias (e.g. 0.1) and add them to focus_items.
- For demand_down or supply_up, use negative watchlist_bias (e.g. -0.1).
- If health shows high blocked_sell_rate or high drawdown, reduce aggression or max_position_gp.
Output only valid JSON, no markdown or explanation."""

    user_parts = ["Context card (use events, watchlist, risk_flags to decide):\n"]
    user_parts.append(json.dumps(context_card, indent=2))
    if health:
        user_parts.append("\n\nOptional health snapshot (use to tighten if bad):\n")
        user_parts.append(json.dumps(health, indent=2))
    user_parts.append("\n\nOutput the risk_config JSON only:")
    return system, "".join(user_parts)


def build_risk_config_llm(
    context_card: Dict[str, Any],
    health: Optional[Dict[str, Any]] = None,
    *,
    api_key: str,
    model: str = "gpt-4o-mini",
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Call LLM to produce a risk_config, then clamp all values to safe bounds.
    """
    system, user = _build_prompt(context_card, health)
    raw = _call_llm(system, user, api_key=api_key, model=model, base_url=base_url)
    out = _parse_json_from_response(raw)

    # Apply same bounds as rule-based manager
    risk_mode = (out.get("risk_mode") or "normal").strip().lower()
    if risk_mode not in ("normal", "cautious"):
        risk_mode = "normal"

    max_position_gp = clamp(out.get("max_position_gp", 1_000_000.0), *MAX_POSITION_GP_RANGE)
    max_units_per_trade = clamp(out.get("max_units_per_trade", 250.0), *MAX_UNITS_PER_TRADE_RANGE)
    max_open_orders = int(clamp(out.get("max_open_orders", 2), *MAX_OPEN_ORDERS_RANGE))
    spread_guard_pct = clamp(out.get("spread_guard_pct", 0.02), *SPREAD_GUARD_PCT_RANGE)
    aggression = clamp(out.get("aggression", 0.7), *AGGRESSION_RANGE)

    watchlist_bias_raw = out.get("watchlist_bias")
    if not isinstance(watchlist_bias_raw, dict):
        watchlist_bias_raw = {}
    watchlist_bias: Dict[str, float] = {}
    for k, v in watchlist_bias_raw.items():
        try:
            watchlist_bias[str(k)] = clamp(float(v), *WATCHLIST_BIAS_RANGE)
        except (TypeError, ValueError):
            pass

    focus_items_raw = out.get("focus_items")
    if not isinstance(focus_items_raw, list):
        focus_items_raw = []
    focus_items: List[int] = []
    for x in focus_items_raw:
        try:
            focus_items.append(int(x))
        except (TypeError, ValueError):
            pass

    return {
        "risk_mode": risk_mode,
        "max_position_gp": max_position_gp,
        "max_units_per_trade": max_units_per_trade,
        "max_open_orders": max_open_orders,
        "spread_guard_pct": spread_guard_pct,
        "aggression": aggression,
        "watchlist_bias": watchlist_bias,
        "focus_items": focus_items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build risk_config from context card using an LLM; output risk_config_llm_<date>.json."
    )
    parser.add_argument("--date", type=str, default=None, help="Date YYYY-MM-DD. Default: today UTC.")
    parser.add_argument("--context-card", type=str, default=None, help="Path to context card JSON (overrides --date).")
    parser.add_argument("--health", type=str, default=None, help="Path to health snapshot JSON (optional).")
    parser.add_argument("--out", type=str, default=None, help="Output path (default: data/news/derived/risk_config_llm_<date>.json).")
    parser.add_argument("--model", type=str, default=None, help="OpenAI model (default: gpt-4o-mini).")
    parser.add_argument("--base-url", type=str, default=None, help="OpenAI-compatible API base URL (optional).")
    args = parser.parse_args()

    import datetime as dt

    if args.date:
        date_str = args.date
    else:
        date_str = dt.datetime.now(dt.timezone.utc).date().isoformat()

    if args.context_card:
        card_path = Path(args.context_card)
    else:
        card_path = DERIVED_ROOT / f"context_card_{date_str}.json"
    if not card_path.exists():
        raise FileNotFoundError(f"Context card not found: {card_path}. Run newsExtract first.")

    with card_path.open("r", encoding="utf-8") as f:
        context_card = json.load(f)

    health: Optional[Dict[str, Any]] = None
    if args.health:
        hp = Path(args.health)
        if hp.exists():
            with hp.open("r", encoding="utf-8") as f:
                health = json.load(f)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in the environment to use the LLM risk manager.")

    model = args.model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")

    risk_config = build_risk_config_llm(
        context_card,
        health,
        api_key=api_key,
        model=model,
        base_url=base_url,
    )

    out_path = Path(args.out) if args.out else DERIVED_ROOT / f"risk_config_llm_{date_str}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(risk_config, f, indent=2)
    print(f"Wrote {out_path}")
    print(
        f"  risk_mode={risk_config['risk_mode']}  max_position_gp={risk_config['max_position_gp']}  "
        f"aggression={risk_config['aggression']}  spread_guard_pct={risk_config['spread_guard_pct']}"
    )


if __name__ == "__main__":
    main()
