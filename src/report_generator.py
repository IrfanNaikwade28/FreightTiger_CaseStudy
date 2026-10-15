"""Deterministic CSV report generation for the FreightTiger case study.

Joins route-week metrics, anomaly results, and evidence verdicts into the
exact output contract of ``data/raw/sample_output_format_v2.csv``:

    route, week_of, cost_per_tonne_km, vs_own_history, vs_similar_routes,
    flagged, matched_note_id, reason

Formatting follows the template: cost per tonne-km with 2 decimals;
percentages signed with 1 decimal and the template's fixed suffixes;
missing baselines render as ``N/A`` plus the same suffix (never zero).

``flagged`` interpretation (template-ambiguous; documented here):
* ``Yes``            -- detector flagged the week and evidence does not
  justify the increase (unexplained).
* ``No (justified)`` -- detector flagged the week and a note that applies
  to this route and week explains the increase, clearing the alert
  (template row 2).
* ``No``             -- detector did not flag the week, regardless of
  whether a context note applies.

The anomaly detector itself never sees justification: flagging is
computed independently, and evidence only chooses between ``Yes`` and
``No (justified)`` on detector-flagged rows. ``matched_note_id`` is
populated only for rows that are both detector-flagged and justified
(the "justified anomaly" the brief describes); it is blank on every
other row, including normal weeks where a note merely coincides. For
flagged rows the ``reason`` quotes the evidence (matched note or why
none applies); non-flagged rows get a fixed no-anomaly sentence and
never quote notes. An optional ``reason_overrides`` mapping may replace
``reason`` text on flagged+justified rows only (used for grounded LLM
rewrites); flags, numbers, and ``matched_note_id`` are never affected
by overrides.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from anomaly_detection import AnomalyResult
from cost_analysis import RouteWeekMetrics
from note_retriever import EvidenceVerdict

OUTPUT_COLUMNS: tuple[str, ...] = (
    "route",
    "week_of",
    "cost_per_tonne_km",
    "vs_own_history",
    "vs_similar_routes",
    "flagged",
    "matched_note_id",
    "reason",
)

DEFAULT_TEMPLATE_PATH = Path("data/raw/sample_output_format_v2.csv")
DEFAULT_OUTPUT_PATH = Path("output/output.csv")

FLAG_YES = "Yes"
FLAG_NO = "No"
FLAG_JUSTIFIED = "No (justified)"

MISSING_BASELINE = "N/A"
OWN_HISTORY_SUFFIX = " vs this route's past average"
SIMILAR_ROUTES_SUFFIX = " vs similar-length routes this week"

NO_ANOMALY_REASON = (
    "No anomaly detected; weekly cost per tonne-km is within the expected "
    "range versus own history and similar-length routes this week."
)


class ReportFormatError(ValueError):
    """Raised when the report cannot honour the output contract."""


@dataclass(frozen=True)
class ReportRow:
    """One fully formatted output row keyed for audit."""

    route: str
    week_of: str
    cost_per_tonne_km: str
    vs_own_history: str
    vs_similar_routes: str
    flagged: str
    matched_note_id: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {column: getattr(self, column) for column in OUTPUT_COLUMNS}


def build_rows(
    metrics: Sequence[RouteWeekMetrics],
    anomalies: Sequence[AnomalyResult],
    verdicts: Sequence[EvidenceVerdict],
    reason_overrides: dict[tuple[str, object], str] | None = None,
) -> list[ReportRow]:
    """Join metrics, anomaly results, and verdicts into formatted rows.

    Inputs are aligned by ``(route, week_of)``; any missing or duplicate
    alignment raises ``ReportFormatError``. One row is produced per
    route-week metric, in the metrics' deterministic order.

    ``reason_overrides`` may supply replacement ``reason`` text (e.g. an
    LLM rewrite) keyed by ``(route, week_of)``. Overrides are applied
    only to rows that are both detector-flagged and evidence-justified,
    and never affect flags, numbers, or ``matched_note_id``.
    """
    anomaly_by_key: dict[tuple[str, object], AnomalyResult] = {}
    for a in anomalies:
        key = (a.metrics.route, a.metrics.week_of)
        if key in anomaly_by_key:
            raise ReportFormatError(f"duplicate anomaly result for {key}")
        anomaly_by_key[key] = a
    verdict_by_key: dict[tuple[str, object], EvidenceVerdict] = {}
    for v in verdicts:
        key = (v.route, v.week_of)
        if key in verdict_by_key:
            raise ReportFormatError(f"duplicate evidence verdict for {key}")
        verdict_by_key[key] = v

    rows: list[ReportRow] = []
    seen: set[tuple[str, object]] = set()
    for metric in metrics:
        key = (metric.route, metric.week_of)
        if key in seen:
            raise ReportFormatError(f"duplicate route-week metric for {key}")
        seen.add(key)
        if key not in anomaly_by_key:
            raise ReportFormatError(f"missing anomaly result for {key}")
        if key not in verdict_by_key:
            raise ReportFormatError(f"missing evidence verdict for {key}")
        anomaly = anomaly_by_key[key]
        verdict = verdict_by_key[key]
        override = reason_overrides.get(key) if reason_overrides else None
        rows.append(
            _format_row(metric, anomaly, verdict, reason_override=override)
        )
    return rows


def write_output(rows: Sequence[ReportRow], path: str | Path) -> Path:
    """Write ``rows`` to ``path`` as CSV, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(OUTPUT_COLUMNS)
        for row in rows:
            writer.writerow([getattr(row, column) for column in OUTPUT_COLUMNS])
    return path


def validate_columns_against_template(template_path: str | Path) -> None:
    """Check the template header matches ``OUTPUT_COLUMNS`` exactly."""
    template_path = Path(template_path)
    if not template_path.is_file():
        raise ReportFormatError(f"template file not found: {template_path}")
    with template_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
    if header is None:
        raise ReportFormatError(f"{template_path}: template has no header row")
    if tuple(header) != OUTPUT_COLUMNS:
        raise ReportFormatError(
            f"{template_path}: header mismatch; expected {OUTPUT_COLUMNS}, "
            f"got {tuple(header)}"
        )


def generate_report(
    metrics: Sequence[RouteWeekMetrics],
    anomalies: Sequence[AnomalyResult],
    verdicts: Sequence[EvidenceVerdict],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    template_path: str | Path = DEFAULT_TEMPLATE_PATH,
    reason_overrides: dict[tuple[str, object], str] | None = None,
) -> list[ReportRow]:
    """Validate the template contract, build rows, and write the CSV."""
    validate_columns_against_template(template_path)
    rows = build_rows(metrics, anomalies, verdicts, reason_overrides)
    write_output(rows, output_path)
    return rows


def _format_row(
    metric: RouteWeekMetrics,
    anomaly: AnomalyResult,
    verdict: EvidenceVerdict,
    reason_override: str | None = None,
) -> ReportRow:
    flagged_cell = _flagged_cell(anomaly.flagged, verdict.justified)
    alert_cleared = anomaly.flagged and verdict.justified
    matched_note_id = verdict.matched_note_id if alert_cleared else ""
    if anomaly.flagged:
        reason = (
            reason_override
            if alert_cleared and reason_override
            else verdict.explanation
        )
    else:
        reason = NO_ANOMALY_REASON
    return ReportRow(
        route=metric.route,
        week_of=metric.week_of.isoformat(),
        cost_per_tonne_km=f"{metric.cost_per_tonne_km:.2f}",
        vs_own_history=_format_pct(
            metric.pct_vs_own_history, OWN_HISTORY_SUFFIX
        ),
        vs_similar_routes=_format_pct(
            metric.pct_vs_similar_routes, SIMILAR_ROUTES_SUFFIX
        ),
        flagged=flagged_cell,
        matched_note_id=matched_note_id,
        reason=reason,
    )


def _flagged_cell(detector_flagged: bool, justified: bool) -> str:
    if not detector_flagged:
        return FLAG_NO
    if justified:
        return FLAG_JUSTIFIED
    return FLAG_YES


def _format_pct(value: float | None, suffix: str) -> str:
    if value is None:
        return MISSING_BASELINE + suffix
    return f"{value:+.1f}%" + suffix
