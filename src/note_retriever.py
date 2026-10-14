"""Deterministic context-note retrieval and evidence verdicts.

Loads ``data/raw/context_notes.csv`` (columns: note_id, date, applies_to,
note) and, for a given route-week, decides whether some note genuinely
explains a cost increase.

Matching is strictly deterministic and rule-based:

* Scope: ``applies_to`` must equal the route name (``Origin-Destination``)
  or be exactly ``All Routes``.
* Period: parsed only from what the note states -- an explicit
  ``from <Month D> to <Month D>`` range, ``starting this week`` (anchor
  week only, because no end is stated and none is invented), ``this
  quarter`` (the calendar quarter of the note date), otherwise the note's
  own Monday-based week. An unparseable range falls back to the note's
  week and is marked ambiguous.
* Impact: keyword classification into ``cost_increase``, ``no_impact``
  (explicit statements that costs/rates/disruptions did not change), or
  ``non_increase`` (recovery / improvement news). Only ``cost_increase``
  notes can justify a rise; ``no_impact`` notes never do.

No LLM is used. Flagging decisions are out of scope here: this module
never changes whether a route-week is flagged.
"""

from __future__ import annotations

import csv
import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from cost_analysis import RouteWeekMetrics

REQUIRED_NOTE_COLUMNS: tuple[str, ...] = ("note_id", "date", "applies_to", "note")

ALL_ROUTES_SCOPE = "All Routes"

SCOPE_ALL_ROUTES = "all_routes"
SCOPE_ROUTE = "route"

IMPACT_COST_INCREASE = "cost_increase"
IMPACT_NO_IMPACT = "no_impact"
IMPACT_NON_INCREASE = "non_increase"

PERIOD_EXPLICIT_RANGE = "explicit_range"
PERIOD_OPEN_ANCHOR_WEEK = "open_ended_anchor_week"
PERIOD_QUARTER = "quarter"
PERIOD_SINGLE_WEEK = "single_week"
PERIOD_FALLBACK_WEEK = "fallback_week"

NO_IMPACT_PHRASES: tuple[str, ...] = (
    "costs were not significantly affected",
    "without a rate change",
    "not part of this dataset",
    "no major disruptions",
    "no significant disruptions",
    "remained normal",
)

COST_INCREASE_PHRASES: tuple[str, ...] = (
    "higher trip costs",
    "surcharge",
    "pushing up",
    "prices rose",
)

NON_INCREASE_PHRASES: tuple[str, ...] = (
    "improved road conditions",
    "returned to normal",
)

_MONTHS: dict[str, int] = {}
for _index, _name in enumerate(
    (
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ),
    start=1,
):
    _MONTHS[_name] = _index
    _MONTHS[_name[:3]] = _index

_EXPLICIT_RANGE_RE = re.compile(
    r"from\s+([A-Za-z]+\s+\d{1,2})\s+to\s+([A-Za-z]+\s+\d{1,2})"
)


class NoteDataError(ValueError):
    """Raised when context-note input fails validation."""


@dataclass(frozen=True)
class ContextNote:
    """A single validated context note."""

    note_id: str
    date: dt.date
    applies_to: str
    text: str


@dataclass(frozen=True)
class NoteEvidence:
    """A note with its parsed scope, impact, and effective period."""

    note: ContextNote
    scope: str
    route_name: str | None
    impact: str
    period_start: dt.date
    period_end: dt.date | None
    period_kind: str
    period_is_ambiguous: bool

    @property
    def note_id(self) -> str:
        return self.note.note_id

    def applies_to_route(self, route: str) -> bool:
        if self.scope == SCOPE_ALL_ROUTES:
            return True
        return self.route_name == route

    def applies_to_week(self, week_of: dt.date) -> bool:
        if week_of < self.period_start:
            return False
        if self.period_end is not None and week_of > self.period_end:
            return False
        return True

    def period_description(self) -> str:
        end = self.period_end.isoformat() if self.period_end else "open-ended"
        return f"{self.period_start.isoformat()} to {end}"


@dataclass(frozen=True)
class EvidenceVerdict:
    """Deterministic evidence outcome for one route-week."""

    route: str
    week_of: dt.date
    justified: bool
    matched_note_id: str | None
    explanation: str
    applicable_note_ids: tuple[str, ...]
    supporting_note_ids: tuple[str, ...]
    rejected_note_ids: tuple[str, ...]


def load_notes(path: str | Path) -> list[ContextNote]:
    """Load and validate context notes from ``path``.

    Raises ``NoteDataError`` on a missing file, missing columns, empty
    input, malformed dates, duplicate ids, or empty required fields.
    """
    path = Path(path)
    if not path.is_file():
        raise NoteDataError(f"notes file not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise NoteDataError(f"{path}: missing header row")
        missing = [c for c in REQUIRED_NOTE_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise NoteDataError(
                f"{path}: missing required columns: {', '.join(missing)}"
            )
        notes = [
            _parse_note_row(path, row_number, row)
            for row_number, row in enumerate(reader, start=2)
        ]
    if not notes:
        raise NoteDataError(f"{path}: no notes found")
    seen: set[str] = set()
    for note in notes:
        if note.note_id in seen:
            raise NoteDataError(f"{path}: duplicate note_id {note.note_id!r}")
        seen.add(note.note_id)
    return notes


def build_evidence(notes: Sequence[ContextNote]) -> list[NoteEvidence]:
    """Classify notes into scope, impact, and effective-period evidence."""
    return [_classify_note(note) for note in notes]


def find_evidence(
    evidences: Sequence[NoteEvidence],
    route: str,
    week_of: dt.date,
) -> EvidenceVerdict:
    """Return the evidence verdict for ``route`` in week ``week_of``.

    A verdict is justified only when at least one applicable note is
    classified as a cost increase for that route and week. No-impact and
    non-increase notes are recorded as rejected and never justify a rise.
    If several supporting notes apply, the most specific and most recent
    wins deterministically (route scope first, then latest period start,
    then note_id); corroborating ids are also reported.
    """
    applicable = [
        e for e in evidences if e.applies_to_route(route) and e.applies_to_week(week_of)
    ]
    supporting = [e for e in applicable if e.impact == IMPACT_COST_INCREASE]
    rejected = [e for e in applicable if e.impact != IMPACT_COST_INCREASE]
    primary = _primary_support(supporting)
    corroborating = tuple(
        e.note_id for e in _sorted_support(supporting) if primary and e is not primary
    )
    if primary is not None:
        explanation = _justified_explanation(primary, route, week_of, corroborating)
        return EvidenceVerdict(
            route=route,
            week_of=week_of,
            justified=True,
            matched_note_id=primary.note_id,
            explanation=explanation,
            applicable_note_ids=tuple(e.note_id for e in applicable),
            supporting_note_ids=tuple(e.note_id for e in _sorted_support(supporting)),
            rejected_note_ids=tuple(e.note_id for e in _sorted_support(rejected)),
        )
    explanation = _unexplained_explanation(evidences, route, week_of, applicable)
    return EvidenceVerdict(
        route=route,
        week_of=week_of,
        justified=False,
        matched_note_id=None,
        explanation=explanation,
        applicable_note_ids=tuple(e.note_id for e in applicable),
        supporting_note_ids=(),
        rejected_note_ids=tuple(e.note_id for e in _sorted_support(rejected)),
    )


def evaluate_route_weeks(
    metrics: Sequence[RouteWeekMetrics],
    evidences: Sequence[NoteEvidence],
) -> list[EvidenceVerdict]:
    """Return an evidence verdict per route-week, preserving input order."""
    return [
        find_evidence(evidences, m.route, m.week_of)
        for m in metrics
    ]


def _parse_note_row(
    path: Path, row_number: int, row: dict[str, str]
) -> ContextNote:
    def where(field: str) -> str:
        return f"{path}: row {row_number}: field '{field}'"

    text = {key: (row.get(key) or "").strip() for key in REQUIRED_NOTE_COLUMNS}
    for field in REQUIRED_NOTE_COLUMNS:
        if not text[field]:
            raise NoteDataError(where(field) + " must not be empty")
    try:
        note_date = dt.date.fromisoformat(text["date"])
    except ValueError as exc:
        raise NoteDataError(
            where("date") + f" is not a valid ISO date: {text['date']!r}"
        ) from exc
    return ContextNote(
        note_id=text["note_id"],
        date=note_date,
        applies_to=text["applies_to"],
        text=text["note"],
    )


def _classify_note(note: ContextNote) -> NoteEvidence:
    if note.applies_to == ALL_ROUTES_SCOPE:
        scope, route_name = SCOPE_ALL_ROUTES, None
    else:
        scope, route_name = SCOPE_ROUTE, note.applies_to
    impact = _classify_impact(note.text)
    period_start, period_end, period_kind, ambiguous = _parse_period(note)
    return NoteEvidence(
        note=note,
        scope=scope,
        route_name=route_name,
        impact=impact,
        period_start=period_start,
        period_end=period_end,
        period_kind=period_kind,
        period_is_ambiguous=ambiguous,
    )


def _classify_impact(text: str) -> str:
    lowered = text.lower()
    if any(phrase in lowered for phrase in NO_IMPACT_PHRASES):
        return IMPACT_NO_IMPACT
    if any(phrase in lowered for phrase in COST_INCREASE_PHRASES):
        return IMPACT_COST_INCREASE
    if any(phrase in lowered for phrase in NON_INCREASE_PHRASES):
        return IMPACT_NON_INCREASE
    return IMPACT_NON_INCREASE


def _parse_period(
    note: ContextNote,
) -> tuple[dt.date, dt.date | None, str, bool]:
    lowered = note.text.lower()
    week_start = _monday_of(note.date)
    week_end = week_start + dt.timedelta(days=6)
    match = _EXPLICIT_RANGE_RE.search(note.text)
    if match:
        start = _parse_month_day(match.group(1), note.date.year)
        end = _parse_month_day(match.group(2), note.date.year)
        if start is not None and end is not None and start <= end:
            return start, end, PERIOD_EXPLICIT_RANGE, False
        return week_start, week_end, PERIOD_FALLBACK_WEEK, True
    if "starting this week" in lowered:
        # No end is stated; conservatively anchor to the note's own week
        # rather than assuming indefinite applicability.
        return week_start, week_end, PERIOD_OPEN_ANCHOR_WEEK, True
    if "this quarter" in lowered:
        quarter_start_month = ((note.date.month - 1) // 3) * 3 + 1
        quarter_start = dt.date(note.date.year, quarter_start_month, 1)
        next_quarter_month = quarter_start_month + 3
        if next_quarter_month > 12:
            quarter_end = dt.date(note.date.year, 12, 31)
        else:
            quarter_end = dt.date(
                note.date.year, next_quarter_month, 1
            ) - dt.timedelta(days=1)
        return quarter_start, quarter_end, PERIOD_QUARTER, False
    return week_start, week_end, PERIOD_SINGLE_WEEK, False


def _parse_month_day(text: str, default_year: int) -> dt.date | None:
    parts = text.strip().split()
    if len(parts) != 2:
        return None
    month = _MONTHS.get(parts[0][:1].upper() + parts[0][1:].lower())
    if month is None:
        return None
    try:
        day = int(parts[1])
        return dt.date(default_year, month, day)
    except ValueError:
        return None


def _monday_of(date: dt.date) -> dt.date:
    return date - dt.timedelta(days=date.weekday())


def _sorted_support(evidences: Sequence[NoteEvidence]) -> list[NoteEvidence]:
    return sorted(
        evidences,
        key=lambda e: (
            0 if e.scope == SCOPE_ROUTE else 1,
            -e.period_start.toordinal(),
            e.note_id,
        ),
    )


def _primary_support(
    supporting: Sequence[NoteEvidence],
) -> NoteEvidence | None:
    if not supporting:
        return None
    return _sorted_support(supporting)[0]


def _justified_explanation(
    primary: NoteEvidence,
    route: str,
    week_of: dt.date,
    corroborating: tuple[str, ...],
) -> str:
    text = primary.note.text.rstrip()
    scope = (
        "all routes"
        if primary.scope == SCOPE_ALL_ROUTES
        else primary.route_name
    )
    explanation = (
        f"Matches note {primary.note_id} dated {primary.note.date.isoformat()} "
        f"for {scope}, effective {primary.period_description()}: {text} "
        "This note applies to this route and week, so the cost rise is explained."
    )
    if corroborating:
        explanation += f" Corroborated by note(s) {', '.join(corroborating)}."
    return explanation


def _unexplained_explanation(
    evidences: Sequence[NoteEvidence],
    route: str,
    week_of: dt.date,
    applicable: Sequence[NoteEvidence],
) -> str:
    if applicable:
        first = _sorted_support(applicable)[0]
        if first.impact == IMPACT_NO_IMPACT:
            return (
                f"No supporting note found. Applicable note {first.note_id} "
                f"dated {first.note.date.isoformat()} for "
                f"{first.note.applies_to} states: {first.note.text.rstrip()} "
                "This reports no cost impact and cannot justify the increase, "
                "so the cost rise is unexplained."
            )
        return (
            f"No supporting note found. Applicable note {first.note_id} "
            f"dated {first.note.date.isoformat()} for "
            f"{first.note.applies_to} states: {first.note.text.rstrip()} "
            "This does not describe a cost increase, so the cost rise is "
            "unexplained."
        )
    supporting_elsewhere = [
        e for e in evidences
        if e.impact == IMPACT_COST_INCREASE
        and (e.applies_to_week(week_of) or e.applies_to_route(route))
    ]
    if supporting_elsewhere:
        first = _sorted_support(supporting_elsewhere)[0]
        if not first.applies_to_route(route):
            return (
                f"Note {first.note_id} dated {first.note.date.isoformat()} "
                f"supports cost rises on {first.note.applies_to}, not on "
                f"{route}; it does not apply here. No matching note found "
                f"for {route} week {week_of.isoformat()}; cost rise is "
                "unexplained."
            )
        return (
            f"Note {first.note_id} dated {first.note.date.isoformat()} for "
            f"{first.note.applies_to} could support a cost rise but its "
            f"effective period {first.period_description()} does not cover "
            f"week {week_of.isoformat()}; cost rise is unexplained."
        )
    return (
        f"No context note found for {route} week {week_of.isoformat()}; "
        "cost rise is unexplained."
    )
