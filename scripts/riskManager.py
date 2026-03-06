"""
Produce a bounded risk_config from context card + optional health snapshot.

Inputs:
  - context_card (events, watchlist, risk_flags from newsExtract)
  - optional health snapshot: blocked_sell_rate, buy_block_rate, drawdown/worst_equity, volatility

Outputs (all clamped):
  - max_position_gp, max_units_per_trade, max_open_orders
  - spread_guard_pct, aggression
  - watchlist_bias: {item_id: float in [-0.25, 0.25]}
  - risk_mode: "normal" | "cautious"
  - focus_items: list of item_ids to bias toward (from watchlist demand_up)

Rules (no LLM):
  - hotfix_rolling or bug_reports → risk_mode=cautious, lower max_position_gp, lower max_open_orders, wider spread_guard_pct
  - high blocked_sell_rate → lower aggression
  - watchlist demand_up → allow higher max_units for those items (or bias); demand_down/supply_up → reduce exposure (bias negative)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
DERIVED_ROOT = ROOT_DIR / "data" / "news" / "derived"

# Bounds (hard limits so config can't do anything wild)
MAX_POSITION_GP_RANGE = (250_000, 2_000_000)
MAX_UNITS_PER_TRADE_RANGE = (1, 500)
MAX_OPEN_ORDERS_RANGE = (1, 3)
SPREAD_GUARD_PCT_RANGE = (0.0, 0.05)
AGGRESSION_RANGE = (0.0, 1.0)
WATCHLIST_BIAS_RANGE = (-0.25, 0.25)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def build_risk_config(
    context_card: Dict[str, Any],
    health: Optional[Dict[str, Any]] = None,
    *,
    cautious_max_position_gp: Optional[float] = None,
    cautious_aggression_cap: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Build bounded risk_config from context card and optional health snapshot.

    health can contain: blocked_sell_rate, buy_block_rate, worst_equity (or drawdown), volatility.
    """
    health = health or {}
    risk_flags: List[str] = context_card.get("risk_flags") or []
    watchlist: List[Dict[str, Any]] = context_card.get("watchlist") or []

    # Defaults (middle of ranges)
    max_position_gp = 1_000_000.0
    max_units_per_trade = 250.0
    max_open_orders = 2
    spread_guard_pct = 0.02
    aggression = 0.7
    risk_mode = "normal"

    # Rule: hotfix or bug_reports → cautious
    if "hotfix_rolling" in risk_flags or "bug_reports" in risk_flags:
        risk_mode = "cautious"
        max_position_gp = cautious_max_position_gp or 500_000.0
        max_open_orders = 2
        spread_guard_pct = 0.045  # sweet spot: balance opportunity vs safety (block when spread_pct > this)

    # Rule: high blocked sell rate → reduce aggression (less aggressive pricing → fewer spam attempts)
    blocked_sell_rate = health.get("blocked_sell_rate")
    if blocked_sell_rate is not None and float(blocked_sell_rate) > 0.2:
        aggression = 0.5
    if risk_mode == "cautious":
        cap = cautious_aggression_cap if cautious_aggression_cap is not None else 0.5
        aggression = min(aggression, cap)

    # Rule: high buy block rate → slightly tighter spread guard (avoid illiquid)
    buy_block_rate = health.get("buy_block_rate")
    if buy_block_rate is not None and float(buy_block_rate) > 0.15:
        spread_guard_pct = min(0.04, spread_guard_pct + 0.01)

    # Rule: bad drawdown/worst_equity → more cautious
    worst_equity = health.get("worst_equity")
    drawdown = health.get("drawdown")
    if worst_equity is not None and float(worst_equity) < 0.9:
        risk_mode = "cautious"
        max_position_gp = min(max_position_gp, 750_000.0)
    if drawdown is not None and float(drawdown) > 0.15:
        max_position_gp = min(max_position_gp, 750_000.0)

    # Watchlist: demand_up → positive bias (allow slightly more); demand_down/supply_up → negative
    watchlist_bias: Dict[int, float] = {}
    focus_items: List[int] = []
    for w in watchlist:
        item_id = int(w.get("item_id", -1))
        if item_id < 0:
            continue
        impact = (w.get("expected_impact") or "").strip().lower()
        if impact == "demand_up":
            watchlist_bias[item_id] = clamp(0.1, *WATCHLIST_BIAS_RANGE)
            focus_items.append(item_id)
        elif impact in ("demand_down", "supply_up"):
            watchlist_bias[item_id] = clamp(-0.1, *WATCHLIST_BIAS_RANGE)

    # Clamp all to bounds
    max_position_gp = clamp(max_position_gp, *MAX_POSITION_GP_RANGE)
    max_units_per_trade = clamp(max_units_per_trade, *MAX_UNITS_PER_TRADE_RANGE)
    max_open_orders = int(clamp(max_open_orders, *MAX_OPEN_ORDERS_RANGE))
    spread_guard_pct = clamp(spread_guard_pct, *SPREAD_GUARD_PCT_RANGE)
    aggression = clamp(aggression, *AGGRESSION_RANGE)
    watchlist_bias = {k: clamp(v, *WATCHLIST_BIAS_RANGE) for k, v in watchlist_bias.items()}

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
    parser = argparse.ArgumentParser(description="Build risk_config from context card (+ optional health).")
    parser.add_argument("--date", type=str, default=None, help="Date YYYY-MM-DD (context_card_<date>.json). Default: today UTC.")
    parser.add_argument("--context-card", type=str, default=None, help="Path to context card JSON (overrides --date).")
    parser.add_argument("--health", type=str, default=None, help="Path to health snapshot JSON (optional).")
    parser.add_argument("--cautious-max-pos-gp", type=float, default=None, help="Override cautious max_position_gp cap (default 500_000).")
    parser.add_argument("--cautious-aggression-cap", type=float, default=None, help="Override aggression cap in cautious mode (default 0.5).")
    parser.add_argument("--out", type=str, default=None, help="Output path for risk_config (default: data/news/derived/risk_config_<date>.json).")
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
        with Path(args.health).open("r", encoding="utf-8") as f:
            health = json.load(f)

    risk_config = build_risk_config(
        context_card,
        health,
        cautious_max_position_gp=args.cautious_max_pos_gp,
        cautious_aggression_cap=args.cautious_aggression_cap,
    )

    out_path = Path(args.out) if args.out else DERIVED_ROOT / f"risk_config_{date_str}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(risk_config, f, indent=2)
    print(f"Wrote {out_path}")
    print(f"  risk_mode={risk_config['risk_mode']}  max_position_gp={risk_config['max_position_gp']}  aggression={risk_config['aggression']}")


if __name__ == "__main__":
    main()
