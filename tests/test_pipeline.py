"""Focused automated tests for the FreightTiger pipeline (stdlib unittest).

Run from the repository root:
    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import datetime as dt
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anomaly_detection import AnomalyResult, detect_anomalies  # noqa: E402
from cost_analysis import (  # noqa: E402
    RouteWeekAggregate,
    RouteWeekMetrics,
    Shipment,
    aggregate_route_weeks,
    analyze_shipments,
    compute_metrics,
    monday_of,
)
from note_retriever import (  # noqa: E402
    EvidenceVerdict,
    build_evidence,
    evaluate_route_weeks,
    find_evidence,
    load_notes,
)
from report_generator import (  # noqa: E402
    MISSING_BASELINE,
    OUTPUT_COLUMNS,
    ReportRow,
    build_rows,
    validate_columns_against_template,
)

SHIPMENTS_CSV = ROOT / "data" / "raw" / "shipment_records.csv"
NOTES_CSV = ROOT / "data" / "raw" / "context_notes.csv"
TEMPLATE_CSV = ROOT / "data" / "raw" / "sample_output_format_v2.csv"

WEEK0 = dt.date(2024, 1, 1)


def _week(offset: int) -> dt.date:
    return WEEK0 + dt.timedelta(weeks=offset)


def _shipment(
    shipment_id: str,
    day: dt.date,
    quantity: float = 10.0,
    distance: float = 100.0,
    cost: float = 1000.0,
    origin: str = "Alpha",
    destination: str = "Beta",
    route_type: str = "Short",
) -> Shipment:
    return Shipment(
        shipment_id=shipment_id,
        origin=origin,
        destination=destination,
        route_type=route_type,
        material="Test",
        quantity_tonnes=quantity,
        distance_km=distance,
        freight_cost_inr=cost,
        shipment_date=day,
        transporter="Test",
    )


def _aggregate(
    origin: str,
    destination: str,
    route_type: str,
    week_of: dt.date,
    cost: float,
    tonne_km: float = 1.0,
) -> RouteWeekAggregate:
    return RouteWeekAggregate(
        origin=origin,
        destination=destination,
        route_type=route_type,
        week_of=week_of,
        total_freight_cost_inr=cost,
        total_tonne_km=tonne_km,
        n_shipments=1,
    )


def _metrics(
    own_pct: float | None,
    peer_pct: float | None,
    week_of: dt.date = WEEK0,
    origin: str = "Alpha",
    destination: str = "Beta",
) -> RouteWeekMetrics:
    return RouteWeekMetrics(
        origin=origin,
        destination=destination,
        route_type="Short",
        week_of=week_of,
        cost_per_tonne_km=1.0,
        pct_vs_own_history=own_pct,
        pct_vs_similar_routes=peer_pct,
        prior_weeks_used=8,
        peer_routes_used=1,
        n_shipments=3,
    )


class TestWeightedWeeklyCost(unittest.TestCase):
    def test_cost_is_total_cost_over_total_tonne_km(self) -> None:
        shipments = [
            _shipment("S1", dt.date(2024, 1, 1), quantity=10, distance=100, cost=1500),
            _shipment("S2", dt.date(2024, 1, 2), quantity=30, distance=100, cost=900),
        ]
        aggregates = aggregate_route_weeks(shipments)
        self.assertEqual(len(aggregates), 1)
        # weighted: 2400 / (10*100 + 30*100) = 0.6; mean-of-ratios would be 0.9
        self.assertAlmostEqual(aggregates[0].cost_per_tonne_km, 0.6)
        self.assertNotAlmostEqual(aggregates[0].cost_per_tonne_km, 0.9)


class TestMondayGrouping(unittest.TestCase):
    def test_monday_of_known_dates(self) -> None:
        self.assertEqual(monday_of(dt.date(2024, 1, 1)), dt.date(2024, 1, 1))
        self.assertEqual(monday_of(dt.date(2024, 1, 3)), dt.date(2024, 1, 1))
        self.assertEqual(monday_of(dt.date(2024, 1, 7)), dt.date(2024, 1, 1))
        self.assertEqual(monday_of(dt.date(2024, 1, 8)), dt.date(2024, 1, 8))

    def test_shipments_on_same_week_share_one_aggregate(self) -> None:
        shipments = [
            _shipment("S1", dt.date(2024, 1, 3)),
            _shipment("S2", dt.date(2024, 1, 7)),
            _shipment("S3", dt.date(2024, 1, 8)),
        ]
        aggregates = aggregate_route_weeks(shipments)
        self.assertEqual(
            sorted(a.week_of for a in aggregates),
            [dt.date(2024, 1, 1), dt.date(2024, 1, 8)],
        )


class TestHistoricalBaseline(unittest.TestCase):
    def test_excludes_current_week_and_uses_at_most_eight_prior(self) -> None:
        # 10 weeks; last week doubles. Baseline must use the 8 weeks
        # before it (all 100), never the current week (which would drag
        # the mean to 110) and never all 9 priors.
        aggregates = [
            _aggregate("A", "B", "Short", _week(i), 100.0 if i < 9 else 200.0)
            for i in range(10)
        ]
        metrics = compute_metrics(aggregates)
        last = metrics[-1]
        self.assertEqual(last.prior_weeks_used, 8)
        self.assertAlmostEqual(last.pct_vs_own_history, 100.0)
        third = metrics[2]
        self.assertEqual(third.prior_weeks_used, 2)
        self.assertAlmostEqual(third.pct_vs_own_history, 0.0)

    def test_first_week_has_missing_history(self) -> None:
        metrics = compute_metrics([_aggregate("A", "B", "Short", WEEK0, 100.0)])
        self.assertIsNone(metrics[0].pct_vs_own_history)
        self.assertEqual(metrics[0].prior_weeks_used, 0)

    def test_short_calendar_gap_skips_missing_week_without_padding(self) -> None:
        aggregates = [
            _aggregate("A", "B", "Short", _week(0), 100.0),
            _aggregate("A", "B", "Short", _week(3), 120.0),
        ]
        metrics = compute_metrics(aggregates)
        self.assertEqual(metrics[1].prior_weeks_used, 1)
        self.assertAlmostEqual(metrics[1].pct_vs_own_history, 20.0)


class TestPeerBaseline(unittest.TestCase):
    def test_peer_mean_excludes_current_route(self) -> None:
        aggregates = [
            _aggregate("A", "B", "Short", WEEK0, 1.0),
            _aggregate("C", "D", "Short", WEEK0, 2.0),
            _aggregate("E", "F", "Short", WEEK0, 4.0),
        ]
        metrics = compute_metrics(aggregates)
        alpha_m = next(m for m in metrics if (m.origin, m.destination) == ("A", "B"))
        self.assertEqual(alpha_m.peer_routes_used, 2)
        self.assertAlmostEqual(alpha_m.pct_vs_similar_routes, (1.0 / 3.0 - 1) * 100)

    def test_no_peers_gives_missing_comparison(self) -> None:
        metrics = compute_metrics([_aggregate("A", "B", "Short", WEEK0, 1.0)])
        self.assertIsNone(metrics[0].pct_vs_similar_routes)
        self.assertEqual(metrics[0].peer_routes_used, 0)


class TestAnomalyThreshold(unittest.TestCase):
    def test_exactly_twenty_percent_is_flagged(self) -> None:
        result = detect_anomalies([_metrics(20.0, 0.0)])[0]
        self.assertTrue(result.flagged)
        self.assertEqual(result.triggers, ("own_history",))

    def test_just_below_twenty_percent_is_not_flagged(self) -> None:
        result = detect_anomalies([_metrics(19.999999, 0.0)])[0]
        self.assertFalse(result.flagged)

    def test_peer_only_flag_and_missing_baselines(self) -> None:
        peer_only = detect_anomalies([_metrics(None, 20.5)])[0]
        self.assertTrue(peer_only.flagged)
        self.assertEqual(peer_only.triggers, ("similar_routes",))
        missing = detect_anomalies([_metrics(None, None)])[0]
        self.assertFalse(missing.flagged)
        below = detect_anomalies([_metrics(-40.0, -5.0)])[0]
        self.assertFalse(below.flagged)


class TestEvidenceMatching(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evidences = build_evidence(load_notes(NOTES_CSV))

    def test_correct_route_and_relevant_date(self) -> None:
        verdict = find_evidence(self.evidences, "Ahmedabad-Mumbai", dt.date(2025, 1, 20))
        self.assertTrue(verdict.justified)
        self.assertEqual(verdict.matched_note_id, "N002")

    def test_wrong_route_is_not_justified(self) -> None:
        verdict = find_evidence(self.evidences, "Mumbai-Pune", dt.date(2025, 1, 20))
        self.assertFalse(verdict.justified)
        self.assertIsNone(verdict.matched_note_id)

    def test_note_outside_effective_period_is_not_justified(self) -> None:
        verdict = find_evidence(self.evidences, "Chennai-Bangalore", dt.date(2024, 6, 4))
        self.assertFalse(verdict.justified)
        verdict = find_evidence(self.evidences, "Chennai-Bangalore", dt.date(2025, 3, 10))
        self.assertFalse(verdict.justified)

    def test_no_impact_notes_never_justify(self) -> None:
        n005 = find_evidence(self.evidences, "Mumbai-Delhi", dt.date(2024, 7, 29))
        self.assertFalse(n005.justified)
        n006 = find_evidence(self.evidences, "Mumbai-Pune", dt.date(2025, 9, 15))
        self.assertFalse(n006.justified)
        n010 = find_evidence(self.evidences, "Delhi-Jaipur", dt.date(2025, 10, 27))
        self.assertFalse(n010.justified)


class TestOutputContract(unittest.TestCase):
    def test_output_columns_match_template_header(self) -> None:
        validate_columns_against_template(TEMPLATE_CSV)
        import csv

        with TEMPLATE_CSV.open(newline="", encoding="utf-8-sig") as handle:
            header = tuple(next(csv.reader(handle)))
        self.assertEqual(header, OUTPUT_COLUMNS)

    def test_missing_historical_baseline_renders_as_na(self) -> None:
        metrics = compute_metrics([_aggregate("A", "B", "Short", WEEK0, 100.0)])
        anomalies = detect_anomalies(metrics)
        verdicts = [
            EvidenceVerdict(
                route=m.route,
                week_of=m.week_of,
                justified=False,
                matched_note_id=None,
                explanation="No context note found.",
                applicable_note_ids=(),
                supporting_note_ids=(),
                rejected_note_ids=(),
            )
            for m in metrics
        ]
        rows = build_rows(metrics, anomalies, verdicts)
        self.assertTrue(rows[0].vs_own_history.startswith(MISSING_BASELINE))
        self.assertNotIn("0.0%", rows[0].vs_own_history)


class TestFlaggedDisplay(unittest.TestCase):
    """Detector flag and evidence verdict map to the flagged cell."""

    @staticmethod
    def _row(detector_flagged: bool, justified: bool) -> ReportRow:
        metric = _metrics(30.0 if detector_flagged else 5.0, None)
        anomaly = AnomalyResult(
            metrics=metric,
            flagged=detector_flagged,
            triggers=("own_history",) if detector_flagged else (),
            own_history_pct=metric.pct_vs_own_history,
            similar_routes_pct=None,
        )
        verdict = EvidenceVerdict(
            route=metric.route,
            week_of=metric.week_of,
            justified=justified,
            matched_note_id="N002" if justified else None,
            explanation=(
                "Matches note N002 dated 2025-01-20 for Ahmedabad-Mumbai."
                if justified
                else "No supporting note found; cost rise is unexplained."
            ),
            applicable_note_ids=("N002",) if justified else (),
            supporting_note_ids=("N002",) if justified else (),
            rejected_note_ids=(),
        )
        rows = build_rows([metric], [anomaly], [verdict])
        return rows[0]

    def test_detector_flagged_and_justified_shows_justified(self) -> None:
        row = self._row(detector_flagged=True, justified=True)
        self.assertEqual(row.flagged, "No (justified)")
        self.assertEqual(row.matched_note_id, "N002")

    def test_detector_flagged_and_unjustified_shows_yes(self) -> None:
        row = self._row(detector_flagged=True, justified=False)
        self.assertEqual(row.flagged, "Yes")
        self.assertEqual(row.matched_note_id, "")

    def test_not_flagged_shows_no_and_blank_note_id_even_when_note_applies(
        self,
    ) -> None:
        row = self._row(detector_flagged=False, justified=True)
        self.assertEqual(row.flagged, "No")
        self.assertEqual(row.matched_note_id, "")
        self.assertNotIn("N002", row.reason)

    def test_note_id_only_on_flagged_and_justified_rows(self) -> None:
        cleared = self._row(detector_flagged=True, justified=True)
        self.assertEqual(cleared.matched_note_id, "N002")
        self.assertIn("N002", cleared.reason)
        unexplained = self._row(detector_flagged=True, justified=False)
        normal_with_note = self._row(detector_flagged=False, justified=True)
        normal_plain = self._row(detector_flagged=False, justified=False)
        self.assertEqual(unexplained.matched_note_id, "")
        self.assertEqual(normal_with_note.matched_note_id, "")
        self.assertEqual(normal_plain.matched_note_id, "")

    def test_not_flagged_and_unjustified_shows_no(self) -> None:
        row = self._row(detector_flagged=False, justified=False)
        self.assertEqual(row.flagged, "No")


class TestDeterminism(unittest.TestCase):
    def test_repeated_analysis_produces_identical_results(self) -> None:
        first = analyze_shipments(SHIPMENTS_CSV)
        second = analyze_shipments(SHIPMENTS_CSV)
        self.assertEqual(first, second)
        self.assertEqual(detect_anomalies(first), detect_anomalies(second))
        evidences = build_evidence(load_notes(NOTES_CSV))
        self.assertEqual(
            evaluate_route_weeks(first, evidences),
            evaluate_route_weeks(second, evidences),
        )

    def test_full_row_generation_is_identical(self) -> None:
        metrics = analyze_shipments(SHIPMENTS_CSV)
        anomalies = detect_anomalies(metrics)
        verdicts = evaluate_route_weeks(
            metrics, build_evidence(load_notes(NOTES_CSV))
        )
        self.assertEqual(build_rows(metrics, anomalies, verdicts), build_rows(metrics, anomalies, verdicts))
        self.assertEqual(len(metrics), 728)


if __name__ == "__main__":
    unittest.main()
