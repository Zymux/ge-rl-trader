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
from scripts.riskManager import build_risk_config as build_risk_config_rule_based

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
        # For OpenAI 4.1+ style models: ask for strict JSON to avoid parse errors.
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    choice = data.get("choices") or []
    if not choice:
        raise ValueError("No choices in LLM response")
    msg = choice[0].get("message") or {}
    content = (msg.get("content") or "").strip()
    return content


def _build_prompt(context_card: Dict[str, Any], health: Optional[Dict[str, Any]], rule_config: Dict[str, Any]) -> tuple[str, str]:
    system = """You are a risk manager for a trading simulator. Given:
- a context card (news events, watchlist, risk flags)
- optional health metrics
- and a reference rule-based risk_config

you must output a SINGLE FLAT JSON object with EXACTLY these six keys (no others):

{
  "risk_mode": "normal" | "cautious",
  "max_position_gp": <number>,
  "max_units_per_trade": <number>,
  "max_open_orders": <integer 1-3>,
  "spread_guard_pct": <number between 0 and 0.05>,
  "aggression": <number between 0 and 1>
}

Constraints:
- Do not include watchlist_bias or focus_items. Only the six keys above (the system will re-use the rule-based watchlist_bias / focus_items).
- All values must be valid JSON (no comments, no trailing commas).

Heuristics:
- Start from the provided rule-based config as a good reference; adjust it instead of inventing something unrelated.
- If risk_flags include "hotfix_rolling" or "bug_reports", use "cautious" mode and lower max_position_gp (e.g. 500000) and a safer spread_guard_pct (e.g. 0.04–0.045).
- If health shows high blocked_sell_rate or high drawdown, reduce aggression or max_position_gp modestly (do not collapse them to near-zero unless the situation is extreme).
- When risk_flags are mild and health is reasonable, avoid being overly conservative: keep max_position_gp and aggression in the general neighborhood of the rule-based values.
- Soft target: keep drawdown under the rule-based level, but avoid reducing turnover so much that the agent almost never trades (i.e., do not shrink both max_position_gp and aggression at the same time by more than ~50% unless risk_flags are very severe).
Return ONLY the JSON object, nothing else."""

    user_parts = ["Context card (events, watchlist, risk_flags, health):\n"]
    user_parts.append(json.dumps(context_card, indent=2))
    if health:
        user_parts.append("\n\nOptional health snapshot (use to tighten if bad):\n")
        user_parts.append(json.dumps(health, indent=2))
    user_parts.append("\n\nReference rule-based risk_config (good baseline to adjust from):\n")
    user_parts.append(json.dumps(rule_config, indent=2))
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
    On any API / parsing error, fall back to the rule-based risk manager.

    Returns a dict that always includes an internal '_source' key:
    - '_source' == 'llm' when the LLM response was used
    - '_source' == 'rule_based_fallback' when we fell back
    """
    # Build the rule-based config first so the LLM can see a sensible reference,
    # and so we can re-use its watchlist_bias / focus_items.
    rule_based_cfg = build_risk_config_rule_based(context_card, health)

    system, user = _build_prompt(context_card, health, rule_based_cfg)
    try:
        raw = _call_llm(system, user, api_key=api_key, model=model, base_url=base_url)
        # Log raw response for debugging
        try:
            dbg_path = DERIVED_ROOT / f"risk_config_llm_{context_card.get('date', 'unknown')}_raw.txt"
            dbg_path.write_text(raw, encoding="utf-8")
        except Exception:
            pass
        out = _parse_json_from_response(raw)
    except Exception as e:
        # Fail-safe: delegate to rule-based manager
        print(f"[riskManagerLLM] LLM call failed ({e!r}); falling back to rule-based riskManager.")
        rb = build_risk_config_rule_based(context_card, health)
        rb["_source"] = "rule_based_fallback"
        return rb

    # Apply same bounds as rule-based manager
    risk_mode = (out.get("risk_mode") or "normal").strip().lower()
    if risk_mode not in ("normal", "cautious"):
        risk_mode = "normal"

    max_position_gp = clamp(out.get("max_position_gp", 1_000_000.0), *MAX_POSITION_GP_RANGE)
    max_units_per_trade = clamp(out.get("max_units_per_trade", 250.0), *MAX_UNITS_PER_TRADE_RANGE)
    max_open_orders = int(clamp(out.get("max_open_orders", 2), *MAX_OPEN_ORDERS_RANGE))
    spread_guard_pct = clamp(out.get("spread_guard_pct", 0.02), *SPREAD_GUARD_PCT_RANGE)
    aggression = clamp(out.get("aggression", 0.7), *AGGRESSION_RANGE)

    # For now, keep watchlist_bias / focus_items from the rule-based manager so that
    # demand_up items still get non-empty bias/focus even when the LLM only controls scalars.
    watchlist_bias: Dict[str, float] = dict(rule_based_cfg.get("watchlist_bias") or {})
    focus_items: List[int] = list(rule_based_cfg.get("focus_items") or [])

    return {
        "risk_mode": risk_mode,
        "max_position_gp": max_position_gp,
        "max_units_per_trade": max_units_per_trade,
        "max_open_orders": max_open_orders,
        "spread_guard_pct": spread_guard_pct,
        "aggression": aggression,
        "watchlist_bias": watchlist_bias,
        "focus_items": focus_items,
        "_source": "llm",
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

    # Read the API key from the standard env var; do not hardcode secrets here.
    api_key = os.environ.get("OPENAI_API_KEY")
    model = args.model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")

    if not api_key:
        print("[riskManagerLLM] OPENAI_API_KEY not set; using rule-based riskManager fallback.")
        risk_config = build_risk_config_rule_based(context_card, health)
        source = "rule_based_fallback"
    else:
        risk_config = build_risk_config_llm(
            context_card,
            health,
            api_key=api_key,
            model=model,
            base_url=base_url,
        )
        # build_risk_config_llm annotates its own source
        source = risk_config.pop("_source", "llm")

    out_path = Path(args.out) if args.out else DERIVED_ROOT / f"risk_config_llm_{date_str}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Annotate source for easier debugging / ablations.
    payload = dict(risk_config)
    payload["_source"] = source

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path}  (source={source})")
    print(
        f"  risk_mode={risk_config['risk_mode']}  max_position_gp={risk_config['max_position_gp']}  "
        f"aggression={risk_config['aggression']}  spread_guard_pct={risk_config['spread_guard_pct']}"
    )


if __name__ == "__main__":
    main()
