"""Weekly route cost analysis for the FreightTiger case study.

Loads shipment records, aggregates them by route and Monday-based week,
computes weighted cost per tonne-km, and compares each route-week against:

* its trailing history: mean of up to the 8 immediately preceding calendar
  weeks of the same route (strictly before the current week, no padding);
* its peers: same-week mean cost per tonne-km of all other routes sharing
  the same route_type.

All logic is deterministic and pure; no thresholds, retrieval, or LLM use.
"""

from __future__ import annotations

import csv
import datetime as dt
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REQUIRED_COLUMNS: tuple[str, ...] = (
    "shipment_id",
    "origin",
    "destination",
    "route_type",
    "material",
    "quantity_tonnes",
    "distance_km",
    "freight_cost_inr",
    "shipment_date",
    "transporter",
)

TRAILING_BASELINE_WEEKS = 8

RouteKey = tuple[str, str, str]


class ShipmentDataError(ValueError):
    """Raised when shipment input fails validation."""


@dataclass(frozen=True)
class Shipment:
    """A single validated shipment record."""

    shipment_id: str
    origin: str
    destination: str
    route_type: str
    material: str
    quantity_tonnes: float
    distance_km: float
    freight_cost_inr: float
    shipment_date: dt.date
    transporter: str

    @property
    def route_key(self) -> RouteKey:
        return (self.origin, self.destination, self.route_type)

    @property
    def route(self) -> str:
        return f"{self.origin}-{self.destination}"

    @property
    def week_of(self) -> dt.date:
        return monday_of(self.shipment_date)


@dataclass(frozen=True)
class RouteWeekAggregate:
    """Totals for one route in one Monday-based week."""

    origin: str
    destination: str
    route_type: str
    week_of: dt.date
    total_freight_cost_inr: float
    total_tonne_km: float
    n_shipments: int

    @property
    def route_key(self) -> RouteKey:
        return (self.origin, self.destination, self.route_type)

    @property
    def route(self) -> str:
        return f"{self.origin}-{self.destination}"

    @property
    def cost_per_tonne_km(self) -> float:
        if self.total_tonne_km <= 0:
            raise ShipmentDataError(
                f"non-positive tonne-km total for {self.route} week {self.week_of}"
            )
        return self.total_freight_cost_inr / self.total_tonne_km


@dataclass(frozen=True)
class RouteWeekMetrics:
    """Cost and baseline comparisons for one route-week."""

    origin: str
    destination: str
    route_type: str
    week_of: dt.date
    cost_per_tonne_km: float
    pct_vs_own_history: float | None
    pct_vs_similar_routes: float | None
    prior_weeks_used: int
    peer_routes_used: int
    n_shipments: int

    @property
    def route_key(self) -> RouteKey:
        return (self.origin, self.destination, self.route_type)

    @property
    def route(self) -> str:
        return f"{self.origin}-{self.destination}"


def monday_of(date: dt.date) -> dt.date:
    """Return the Monday of the ISO week containing ``date``."""
    return date - dt.timedelta(days=date.weekday())


def load_shipments(path: str | Path) -> list[Shipment]:
    """Load and validate shipment records from ``path``.

    Raises ``ShipmentDataError`` on a missing file, missing columns,
    empty input, malformed values, or duplicate shipment ids.
    """
    path = Path(path)
    if not path.is_file():
        raise ShipmentDataError(f"shipment file not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ShipmentDataError(f"{path}: missing header row")
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ShipmentDataError(
                f"{path}: missing required columns: {', '.join(missing)}"
            )
        shipments = [
            _parse_row(path, row_number, row)
            for row_number, row in enumerate(reader, start=2)
        ]
    if not shipments:
        raise ShipmentDataError(f"{path}: no shipment records found")
    _check_unique_ids(path, shipments)
    return shipments


def aggregate_route_weeks(
    shipments: Sequence[Shipment],
) -> list[RouteWeekAggregate]:
    """Aggregate shipments into route-week totals, sorted deterministically.

    Weekly cost per tonne-km is later computed as
    sum(freight_cost_inr) / sum(quantity_tonnes * distance_km).
    """
    totals: dict[RouteKey, dict[dt.date, list[float]]] = {}
    for s in shipments:
        week = totals.setdefault(s.route_key, {}).setdefault(
            s.week_of, [0.0, 0.0, 0]
        )
        week[0] += s.freight_cost_inr
        week[1] += s.quantity_tonnes * s.distance_km
        week[2] += 1
    aggregates = [
        RouteWeekAggregate(
            origin=origin,
            destination=destination,
            route_type=route_type,
            week_of=week,
            total_freight_cost_inr=totals_list[0],
            total_tonne_km=totals_list[1],
            n_shipments=int(totals_list[2]),
        )
        for (origin, destination, route_type), weeks in totals.items()
        for week, totals_list in weeks.items()
    ]
    return sorted(
        aggregates,
        key=lambda a: (a.origin, a.destination, a.route_type, a.week_of),
    )


def compute_metrics(
    aggregates: Sequence[RouteWeekAggregate],
) -> list[RouteWeekMetrics]:
    """Compute baseline comparisons for each route-week aggregate.

    Percentage comparisons are ``(current / baseline - 1) * 100``. A
    comparison is ``None`` when its baseline is unavailable or invalid
    (no prior weeks, no peers, or a zero baseline). The trailing-history
    baseline is the mean of up to the ``TRAILING_BASELINE_WEEKS``
    immediately preceding calendar weeks of the same route, excluding the
    current week; missing weeks are skipped, never padded.
    """
    cost_by_route: dict[RouteKey, dict[dt.date, float]] = {}
    cost_by_type_week: dict[str, dict[dt.date, dict[RouteKey, float]]] = {}
    for agg in aggregates:
        cost = agg.cost_per_tonne_km
        cost_by_route.setdefault(agg.route_key, {})[agg.week_of] = cost
        cost_by_type_week.setdefault(agg.route_type, {}).setdefault(
            agg.week_of, {}
        )[agg.route_key] = cost

    metrics: list[RouteWeekMetrics] = []
    for agg in aggregates:
        current = agg.cost_per_tonne_km
        prior_costs = _trailing_week_costs(
            cost_by_route.get(agg.route_key, {}), agg.week_of
        )
        own_baseline = _mean_or_none(prior_costs)
        peer_costs = [
            cost
            for route_key, cost in cost_by_type_week[agg.route_type][agg.week_of].items()
            if route_key != agg.route_key
        ]
        peer_baseline = _mean_or_none(peer_costs)
        metrics.append(
            RouteWeekMetrics(
                origin=agg.origin,
                destination=agg.destination,
                route_type=agg.route_type,
                week_of=agg.week_of,
                cost_per_tonne_km=current,
                pct_vs_own_history=_pct_change(current, own_baseline),
                pct_vs_similar_routes=_pct_change(current, peer_baseline),
                prior_weeks_used=len(prior_costs),
                peer_routes_used=len(peer_costs),
                n_shipments=agg.n_shipments,
            )
        )
    return sorted(
        metrics,
        key=lambda m: (m.origin, m.destination, m.route_type, m.week_of),
    )


def analyze_shipments(path: str | Path) -> list[RouteWeekMetrics]:
    """Load ``path`` and return route-week metrics, deterministically sorted."""
    return compute_metrics(aggregate_route_weeks(load_shipments(path)))


def _parse_row(path: Path, row_number: int, row: dict[str, str]) -> Shipment:
    def where(field: str) -> str:
        return f"{path}: row {row_number}: field '{field}'"

    text = {key: (row.get(key) or "").strip() for key in REQUIRED_COLUMNS}
    for field in ("shipment_id", "origin", "destination", "route_type"):
        if not text[field]:
            raise ShipmentDataError(where(field) + " must not be empty")
    try:
        parsed_date = dt.date.fromisoformat(text["shipment_date"])
    except ValueError as exc:
        raise ShipmentDataError(
            where("shipment_date") + f" is not a valid ISO date: {text['shipment_date']!r}"
        ) from exc
    quantity = _parse_positive_float(text["quantity_tonnes"], where("quantity_tonnes"))
    distance = _parse_positive_float(text["distance_km"], where("distance_km"))
    cost = _parse_nonnegative_float(text["freight_cost_inr"], where("freight_cost_inr"))
    return Shipment(
        shipment_id=text["shipment_id"],
        origin=text["origin"],
        destination=text["destination"],
        route_type=text["route_type"],
        material=text["material"],
        quantity_tonnes=quantity,
        distance_km=distance,
        freight_cost_inr=cost,
        shipment_date=parsed_date,
        transporter=text["transporter"],
    )


def _parse_positive_float(value: str, label: str) -> float:
    parsed = _parse_float(value, label)
    if parsed <= 0:
        raise ShipmentDataError(f"{label} must be > 0, got {parsed}")
    return parsed


def _parse_nonnegative_float(value: str, label: str) -> float:
    parsed = _parse_float(value, label)
    if parsed < 0:
        raise ShipmentDataError(f"{label} must be >= 0, got {parsed}")
    return parsed


def _parse_float(value: str, label: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ShipmentDataError(f"{label} is not numeric: {value!r}") from exc
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        raise ShipmentDataError(f"{label} is not finite: {value!r}")
    return parsed


def _check_unique_ids(path: Path, shipments: Sequence[Shipment]) -> None:
    seen: set[str] = set()
    for s in shipments:
        if s.shipment_id in seen:
            raise ShipmentDataError(
                f"{path}: duplicate shipment_id {s.shipment_id!r}"
            )
        seen.add(s.shipment_id)


def _trailing_week_costs(
    cost_by_week: dict[dt.date, float], week_of: dt.date
) -> list[float]:
    costs = []
    for offset in range(1, TRAILING_BASELINE_WEEKS + 1):
        prior_week = week_of - dt.timedelta(days=7 * offset)
        if prior_week in cost_by_week:
            costs.append(cost_by_week[prior_week])
    return costs


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    mean = statistics.fmean(values)
    return mean if mean > 0 else None


def _pct_change(current: float, baseline: float | None) -> float | None:
    if baseline is None or baseline <= 0:
        return None
    return (current / baseline - 1.0) * 100.0
