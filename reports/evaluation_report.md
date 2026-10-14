# Evaluation Report — FreightTiger Smart Shipping Cost Monitoring Assistant

Date of run: 2026-10-15 · Python 3.14.7 · inputs unchanged (`data/raw/shipment_records.csv`, `data/raw/context_notes.csv`, `data/raw/sample_output_format_v2.csv`)

## 1. Automated tests (`python3 -m unittest discover -s tests -v`)

**24 tests, all passing (OK), ~0.13 s.** Coverage:

| Area | Tests |
|---|---|
| Weighted weekly cost (`Σcost / Σ(qty×dist)`, not mean-of-ratios) | 1 |
| Monday-based week grouping (Mon/Sun boundaries, same-week merge) | 2 |
| Historical baseline (current week excluded, ≤8 prior calendar weeks, no padding on gaps, first week missing) | 3 |
| Peer baseline (self excluded, no peers → missing) | 2 |
| Inclusive +20% anomaly threshold (boundary, sub-threshold, peer-only, missing baselines, below baseline) | 3 |
| Evidence matching (correct route+date, wrong route, out-of-period, no-impact notes never justify) | 4 |
| Output contract (header == template header; missing baseline renders `N/A`, never zero) | 2 |
| Flagged display mapping (flagged+justified → `No (justified)` + note id; flagged+unjustified → `Yes`; not flagged → `No` with blank note id even when a note applies; note ids only on flagged+justified rows) | 5 |
| Determinism (repeat analysis/detection/evidence/row generation identical) | 2 |

## 2. Full-pipeline run (`python3 main.py`)

- Route-weeks processed: **728** (7 routes × 104 weeks)
- Output written: `output/output.csv` (729 lines incl. header), exit code 0

### Detector flags (independent of justification)

- Anomalies detected (threshold: ≥ +20% vs own-history baseline **or** ≥ +20% vs same-week peer baseline): **20**
  - own-history only: 1 · peer only: 14 · both: 5
- Unexplained anomalies: **17**

### Evidence verdicts (independent of flagging)

- Justified route-weeks: **10** — N001 ×2 (Chennai-Bangalore 2025-02-24, 2025-03-03), N002 ×1 (Ahmedabad-Mumbai 2025-01-20), N003 ×7 (all routes, onset week 2025-05-05)
- Of the 20 detector flags, **3** were justified (Ahmedabad-Mumbai 2025-01-20 → N002; Chennai-Bangalore 2025-02-24 and 2025-03-03 → N001); 17 remain unexplained

### Final CSV display values (template vocabulary)

| `flagged` cell | Count | Meaning |
|---|---|---|
| `Yes` | 17 | Detector flagged, no supporting note (unexplained) |
| `No (justified)` | 3 | Detector flagged and a note applies to that route-week explaining the increase (Ahmedabad-Mumbai 2025-01-20 → N002; Chennai-Bangalore 2025-02-24, 2025-03-03 → N001) |
| `No` | 708 | Detector did not flag, regardless of whether a context note applies (includes the 7 normal N003-onset-week rows, which have blank `matched_note_id` and the fixed no-anomaly reason) |

- `matched_note_id` populated on exactly the **3** detector-flagged, justified rows (N001 ×2, N002 ×1); blank on all `Yes` and all `No` rows. Normal weeks that merely coincide with a note (e.g. the seven N003-onset-week rows) show `No` with a blank id and the fixed no-anomaly reason. Thirteen `Yes` rows cite N003 (or another note) in `reason` only as rejected/out-of-window evidence — never as a match.
- Missing historical baselines rendered as `N/A vs this route's past average`: **7** (first week of each route); missing peer baselines: 0

### Template reconciliation (sample rows vs our output)

All three supplied sample rows reproduce exactly with our pipeline (the template's values were computed with the same weighted formula):

| Template row | Template | Our output |
|---|---|---|
| Delhi-Jaipur 2024-11-11 | 4.17, +35.5%, +21.0%, Yes, blank note | identical |
| Ahmedabad-Mumbai 2025-01-20 | 3.29, +29.5%, +22.5%, No (justified), N002 | identical |
| Mumbai-Pune 2025-09-15 | 3.98, +9.2%, +23.6%, Yes, blank note | identical |

## 3. Reproducibility (`python3 reports/run_reproducibility_check.py`)

Three full `main.py` runs against unchanged inputs produced **byte-identical** CSVs
(SHA-256 `394d5aa5c5bf3ce20fb2b8908937ab435761667d87d4ad6cd43eda1d670b698a` for all three runs). **PASS** — see `reports/reproducibility_check.txt`.

## 4. Input validation

- Invalid input (e.g. missing shipments file) exits with code **2** and a clear `Input error:` message; unexpected failures exit with code **1** (verified).

## 5. Honest gaps

- No LLM is used at runtime; explanations are deterministic (see `reports/ai_usage_cost_report.md`).
- The stretch-goal conversational Q&A was not built (not required for the core submission).
- Justification of a route-week does not consider note *magnitude* (e.g. N003's stated 5–7% diesel effect vs a larger spike); it validates route, window, and cost-increase content only.
