"""Deterministic anomaly flagging for route-week cost metrics.

A route-week is flagged when its cost per tonne-km is at least
``ANOMALY_THRESHOLD_PCT`` percent above an *available* baseline, measured
against either:

* the route's trailing own-history baseline, or
* the same-week peer (same route_type) baseline.

Missing baselines are never treated as zero and never trigger a flag by
themselves. Flagging is independent of context-note retrieval: whether a
cost rise is later judged "justified" does not change the flag.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from cost_analysis import RouteWeekMetrics

ANOMALY_THRESHOLD_PCT = 20.0

FLAG_OWN_HISTORY = "own_history"
FLAG_SIMILAR_ROUTES = "similar_routes"


@dataclass(frozen=True)
class AnomalyResult:
    """Flag outcome for one route-week, preserving both comparisons."""

    metrics: RouteWeekMetrics
    flagged: bool
    triggers: tuple[str, ...]
    own_history_pct: float | None
    similar_routes_pct: float | None


def detect_anomalies(
    metrics: Sequence[RouteWeekMetrics],
    threshold_pct: float = ANOMALY_THRESHOLD_PCT,
) -> list[AnomalyResult]:
    """Flag route-weeks whose cost is >= ``threshold_pct``% above a baseline.

    A comparison triggers only when its percentage value is available
    (not ``None``, finite) and is greater than or equal to
    ``threshold_pct``. Unrounded values are compared. Input order is
    preserved, so callers get the same deterministic ordering as
    ``cost_analysis.compute_metrics``.
    """
    if not math.isfinite(threshold_pct) or threshold_pct < 0:
        raise ValueError(
            f"threshold_pct must be a finite value >= 0, got {threshold_pct!r}"
        )
    results = []
    for metric in metrics:
        own_pct = _available_pct(metric.pct_vs_own_history)
        peer_pct = _available_pct(metric.pct_vs_similar_routes)
        triggers = []
        if own_pct is not None and own_pct >= threshold_pct:
            triggers.append(FLAG_OWN_HISTORY)
        if peer_pct is not None and peer_pct >= threshold_pct:
            triggers.append(FLAG_SIMILAR_ROUTES)
        results.append(
            AnomalyResult(
                metrics=metric,
                flagged=bool(triggers),
                triggers=tuple(triggers),
                own_history_pct=own_pct,
                similar_routes_pct=peer_pct,
            )
        )
    return results


def _available_pct(value: float | None) -> float | None:
    """Return ``value`` when it is a usable comparison; else ``None``."""
    if value is None or not math.isfinite(value):
        return None
    return value
