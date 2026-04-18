from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def extract_lqr_costs(policy: Any) -> Optional[dict[str, list[float]]]:
    if hasattr(policy, "get_lqr_costs"):
        return policy.get_lqr_costs()
    action_net = getattr(policy, "action_net", None)
    if action_net is not None and hasattr(action_net, "get_lqr_costs"):
        return action_net.get_lqr_costs()
    return None


def save_lqr_costs(path: Path, policy: Any) -> None:
    costs = extract_lqr_costs(policy)
    if costs is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(costs, fp, indent=2, ensure_ascii=False)
