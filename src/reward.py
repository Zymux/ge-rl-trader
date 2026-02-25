from __future__ import annotations
import numpy as np

def equity_log_reward(prev_worth: float, new_worth: float, eps: float = 1e-12, clip: float | None = 1.0) -> float:
    r = float(np.log(max(new_worth, eps) / max(prev_worth, eps)))
    if clip is not None:
        r = float(np.clip(r, -clip, clip))
    return r