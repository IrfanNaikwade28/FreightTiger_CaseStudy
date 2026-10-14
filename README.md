# FreightTiger Smart Shipping Cost Monitoring Assistant

Deterministic Python system that analyses weekly shipping costs per route, flags unusual increases against two baselines, grounds explanations in supplied context notes, and emits a contract-compliant CSV. Built for the FreightTiger Software Engineer Intern (AI) 24-Hour Challenge.

## Setup

Requires **Python 3.10+** (tested on 3.14). No third-party packages - see `requirements.txt`.

```bash
cd FreightTiger_CaseStudy
python3 -m unittest discover -s tests -v   # run tests
python3 main.py                            # full pipeline
```

## Execution

`python3 main.py` reads the supplied CSVs, writes `output/output.csv`, and prints a summary. Exit codes: `0` success, `2` invalid input, `1` unexpected failure. Paths are overridable:

```bash
python3 main.py --shipments PATH --notes PATH --template PATH --output PATH
```

## Architecture

```
main.py                     CLI entry point; wires modules; summary + exit codes
src/
  cost_analysis.py          Load/validate shipments; weekly aggregation; baselines
  anomaly_detection.py      +20% threshold flagging (independent of justification)
  note_retriever.py         Deterministic note parsing, matching, evidence verdicts
  report_generator.py       Template-exact CSV rows and serialization
tests/test_pipeline.py      24 automated tests (stdlib unittest)
reports/
  run_reproducibility_check.py  3-run byte-comparison harness
  reproducibility_check.txt     Result of the check
  evaluation_report.md          Test + pipeline metrics
  ai_usage_cost_report.md       Honest zero-LLM usage accounting
data/raw/                   Supplied inputs (unmodified)
output/output.csv           Generated deliverable
```

Design principle: calculations, baselines, flags, and evidence validation are pure deterministic Python. Any future LLM layer may only paraphrase already-validated findings; it can never change verdicts or numbers.

## Formulas

Weekly cost per tonne-km (weighted aggregate, not a mean of per-shipment ratios):

```
cost_per_tonne_km(route, week) = sum(freight_cost_inr) / sum(quantity_tonnes * distance_km)
```

over all shipments in that Monday-based week (Monday-Sunday; week_of = that week's Monday).

vs own history: trailing 8-week rolling mean of the same route's weekly cost, using only weeks strictly before the current week (no look-ahead, no padding). With fewer than 8 prior weeks, all available priors are used (reported via prior_weeks_used); the first week of each route has no baseline. Comparison: (current / baseline - 1) * 100.

vs similar routes: same-week mean weekly cost of all other routes sharing the same route_type (Short/Medium/Long), excluding the current route; same percentage formula.

## Threshold choice

The brief fixes the baselines but not a flagging number. We adopt a transparent, documented rule:

Flag a route-week when its cost is >= +20% above an available own-history baseline OR >= +20% above an available same-week peer baseline.

Rationale: (a) the supplied template's own examples behave exactly like a >=20% either-baseline rule and reconcile 3/3 with our computed values; (b) either-baseline is more sensitive than requiring both; (c) inclusive at the boundary is unambiguous. Missing baselines are never treated as zero and never flag by themselves. All comparisons use unrounded values. The constant lives in src/anomaly_detection.py (ANOMALY_THRESHOLD_PCT = 20.0).

## Note-matching rules (src/note_retriever.py)

A note can justify a cost increase only if all of:

1. Scope: applies_to equals the route (Origin-Destination) or is exactly All Routes.
2. Period - parsed only from what the note states, never invented:
   - explicit "from Month D to Month D" range (year from note date) -> that window;
   - "starting this week" -> the note's own week only (no end stated, so no indefinite extrapolation; marked ambiguous);
   - "this quarter" -> the calendar quarter of the note date;
   - otherwise -> the note's Monday-based week.
3. Impact: text classified by keyword rules into cost_increase, no_impact (explicit statements that costs were not affected, no disruptions, no rate change, or not in this dataset), or non_increase (improvement/recovery news). Only cost_increase notes justify. no_impact notes are recorded as rejected evidence and can never justify - a rejected note id and its verbatim text appear in the reason of unexplained rows when relevant.

When several supporting notes apply, precedence is route-scope over All-Routes, then latest period start, then note_id; others are listed as corroborating. Everything is deterministic; no LLM is involved.

## Output schema

output/output.csv columns (exact template contract, validated at runtime against data/raw/sample_output_format_v2.csv):

route, week_of, cost_per_tonne_km, vs_own_history, vs_similar_routes, flagged, matched_note_id, reason

- route: Origin-Destination; week_of: Monday YYYY-MM-DD; cost: 2 decimals.
- Percentages: signed 1 decimal with the template suffixes ("+35.5% vs this route's past average", "+21.0% vs similar-length routes this week"). Missing baselines render N/A plus the same suffix (never zero).
- flagged values (documented interpretation of template ambiguity):
  - Yes - detector flagged and evidence does not justify the increase (unexplained);
  - No (justified) - detector flagged and a note applies to this route-week and explains the cost increase, clearing the alert;
  - No - detector did not flag, regardless of whether a context note applies.
  Detector flags (20) and display cells (17 Yes / 3 No (justified) / 708 No) are distinct; justification never changes the detector.
- matched_note_id: note id only for rows that are both detector-flagged and justified (the brief's "justified" anomaly); blank on every other row, including normal weeks where a context note merely coincides. reason: on flagged rows, a verbatim-grounded explanation (matched note, or exactly why no note applies); non-flagged rows get a fixed no-anomaly sentence and never quote notes. CSV escaping is handled by the standard csv module.

## Testing

python3 -m unittest discover -s tests -v runs 24 tests covering weighted cost vs mean-of-ratios, Monday grouping, baseline windows (exclusion of current week, at most 8 priors, no padding), peer exclusion, +20% inclusivity, evidence route/date matching, no-impact rejection, header contract, N/A handling, the three-case flagged display mapping with note-id population rules, and determinism. Full results are in reports/evaluation_report.md.

## Reproducibility

python3 reports/run_reproducibility_check.py runs main.py three times against unchanged inputs and compares SHA-256 of the CSV bytes; the result is saved to reports/reproducibility_check.txt. Latest result: PASS, all three runs byte-identical. All logic is deterministic (no randomness, no clock dependence, no LLM).

## AI usage

Zero LLM calls at runtime; zero tokens; zero cost. Retrieval and explanations are rule-based and template-generated from note fields - an intentional guardrail-first choice, since the brief warns that a confident wrong reason scores worse than "could not find one." With only 10 notes, exact/rule retrieval is fully auditable and cannot hallucinate. See reports/ai_usage_cost_report.md for the full accounting. An optional LLM paraphrase layer could be added later behind a flag, instrumented for measured token usage, without touching verdicts or numbers.

## Limitations

- Impact classification is keyword-based; it is calibrated to the supplied notes and would need review for a larger or differently worded corpus.
- Note magnitude is not compared to spike size (e.g. the 5-7% diesel note can justify a larger spike if route and window match).
- "starting this week" notes are conservatively applied only to their anchor week.
- A single note id is reported per row; corroborating ids appear only in the explanation text.
- The stretch-goal conversational Q&A was out of scope for the core submission.

## Why no frontend

The brief lists the deliverables as working code, a contract-compliant CSV, a README, a reproducibility check, and a token/cost log. It explicitly states that a web interface, frontend, Django/FastAPI, and a chatbot are not required. A CLI entry point keeps the submission focused, deterministic, and easy to review; a UI would add surface area without improving any graded criterion.
