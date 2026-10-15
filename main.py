#!/usr/bin/env python3
"""FreightTiger Smart Shipping Cost Monitoring Assistant - entry point.

Loads shipment records and context notes, runs cost analysis, anomaly
detection (+20% threshold), and deterministic evidence evaluation, then
writes ``output/output.csv`` in the supplied template format.

Exit codes: 0 success; 2 invalid input; 1 unexpected processing failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from anomaly_detection import detect_anomalies  # noqa: E402
from cost_analysis import ShipmentDataError, analyze_shipments  # noqa: E402
from llm_explainer import DEFAULT_MAX_CALLS, build_explainer  # noqa: E402
from note_retriever import (  # noqa: E402
    NoteDataError,
    build_evidence,
    evaluate_route_weeks,
    load_notes,
)
from report_generator import (  # noqa: E402
    FLAG_JUSTIFIED,
    FLAG_YES,
    ReportFormatError,
    generate_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze weekly shipping costs, flag unusual increases, and "
            "ground explanations in context notes."
        )
    )
    parser.add_argument(
        "--shipments",
        type=Path,
        default=ROOT / "data" / "raw" / "shipment_records.csv",
        help="Path to shipment_records.csv",
    )
    parser.add_argument(
        "--notes",
        type=Path,
        default=ROOT / "data" / "raw" / "context_notes.csv",
        help="Path to context_notes.csv",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=ROOT / "data" / "raw" / "sample_output_format_v2.csv",
        help="Path to the sample output template (header contract)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output" / "output.csv",
        help="Path for the generated output CSV",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help=(
            "Use Groq (OpenAI-compatible API, requires "
            "FREIGHTTIGER_LLM_API_KEY) to rewrite the reason text of "
            "flagged, evidence-justified anomalies. Without a key the "
            "run stays offline and deterministic."
        ),
    )
    parser.add_argument(
        "--llm-max-calls",
        type=int,
        default=DEFAULT_MAX_CALLS,
        help=f"Hard cap on LLM calls per run (default {DEFAULT_MAX_CALLS})",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    metrics = analyze_shipments(args.shipments)
    anomalies = detect_anomalies(metrics)
    evidences = build_evidence(load_notes(args.notes))
    verdicts = evaluate_route_weeks(metrics, evidences)

    verdict_by_key = {(v.route, v.week_of): v for v in verdicts}
    explainer = build_explainer(args.llm, max_calls=args.llm_max_calls)
    reason_overrides: dict[tuple[str, object], str] = {}
    if explainer.config.enabled:
        for anomaly in anomalies:
            if not anomaly.flagged:
                continue
            key = (anomaly.metrics.route, anomaly.metrics.week_of)
            verdict = verdict_by_key[key]
            if not verdict.justified:
                continue
            text = explainer.explain_justified(
                anomaly.metrics, anomaly.flagged, verdict
            )
            if text is not None:
                reason_overrides[key] = text

    rows = generate_report(
        metrics,
        anomalies,
        verdicts,
        output_path=args.output,
        template_path=args.template,
        reason_overrides=reason_overrides,
    )

    detector_flagged = 0
    justified_flagged = 0
    for anomaly in anomalies:
        if not anomaly.flagged:
            continue
        detector_flagged += 1
        key = (anomaly.metrics.route, anomaly.metrics.week_of)
        if verdict_by_key[key].justified:
            justified_flagged += 1
    justified_total = sum(1 for v in verdicts if v.justified)
    unexplained_flagged = detector_flagged - justified_flagged
    display_yes = sum(1 for r in rows if r.flagged == FLAG_YES)
    display_justified = sum(1 for r in rows if r.flagged == FLAG_JUSTIFIED)

    print("FreightTiger shipping cost monitoring assistant")
    print(f"  route-weeks processed : {len(metrics)}")
    print(f"  anomalies detected    : {detector_flagged} (threshold +20%)")
    print(f"  justified findings    : {justified_total} ({justified_flagged} of them anomalies)")
    print(f"  unexplained anomalies : {unexplained_flagged}")
    print(
        f"  display cells         : {display_yes} 'Yes', "
        f"{display_justified} 'No (justified)'"
    )
    usage = explainer.usage
    if usage.mode == "online":
        if usage.estimated_cost_usd is not None:
            cost = f"${usage.estimated_cost_usd:.6f}"
        elif usage.errors and usage.errors == usage.calls:
            cost = "none (no successful calls)"
        else:
            cost = "unknown (model not in pricing table)"
        print(
            f"  llm explanations      : online, model={usage.model}, "
            f"calls={usage.calls}, prompt_tokens={usage.prompt_tokens}, "
            f"completion_tokens={usage.completion_tokens}, est_cost={cost}, "
            f"errors={usage.errors}, fallbacks={usage.fallbacks}"
        )
        if usage.usage_metadata_missing:
            print(
                "  llm usage metadata    : missing on "
                f"{usage.usage_metadata_missing} call(s); token counts "
                "exclude those calls"
            )
        for record in usage.error_records[:5]:
            status = (
                f"http_{record.http_status}"
                if record.http_status is not None
                else record.kind
            )
            ray = f" ray={record.ray_id}" if record.ray_id else ""
            print(
                f"  llm error             : call#{record.call_index} "
                f"{status}{ray}: {record.message}"
            )
        if len(usage.error_records) > 5:
            print(
                f"  llm error             : ... "
                f"{len(usage.error_records) - 5} more not shown"
            )
    elif args.llm:
        print(
            "  llm explanations      : requested but no API key found "
            "(FREIGHTTIGER_LLM_API_KEY); ran offline with deterministic "
            "reasons"
        )
    else:
        print("  llm explanations      : offline (deterministic reasons)")
    print(f"  output written to     : {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (ShipmentDataError, NoteDataError, ReportFormatError) as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"Processing error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
