# Due-date bake-off — 20k, 2026-08-18

Phase 1 of the "fix the due-date model, then relabel 20k" plan. Five candidate due-date models,
scored against actual `Service_Num == 20` completion dates over the same 8-quarter cohort walk
`run_history_mode()` uses in production (`year=2026, quarter=Q1, quarters=8` -> 2024Q1..2025Q4,
grace ±45 days). Implementation: `due_date.py` (new module) + a throwaway driver script (not
committed). Nothing in `models/20k/`, `refactored_test_dir/final_processed_20k*.csv` or
`test_sets/test_20k_Q*_2026.csv` was touched.

## Candidates

| candidate | definition |
|---|---|
| `schedule` | `FirstSrvDate + 6mo x (m/10)` — existing default |
| `calibrated` | `FirstSrvDate + MILESTONE_MONTHS_CALIBRATED[20]` (9mo) — existing, biased early per audit |
| `mileage` | last service date + remaining km / daily mileage rate, cut at the cohort's cutoff. Fixed relative to `create_test_cohort(use_mileage_projection=True)`: a vehicle already past the target mileage, or with <30 days of history, is EXCLUDED, not collapsed onto the last service date |
| `interval` | last completed lower-milestone (10k) date + remaining km / (per-vehicle km-per-block, from the same diff logic as `derive_pms_mileage_features`, divided by the POPULATION median days-per-block measured at that cutoff — with only one prior milestone before 20k, almost no vehicle has a per-vehicle day-interval, so this population median normalizer does most of the work; see caveat below) |
| `earliest` | `min(schedule, mileage)` per vehicle, NaT-safe — "every 10,000km or 6 months, whichever comes first" |

## Verification #1 — does the harness reproduce the known `schedule` numbers?

**Cohort counts: exact match.** Summed over the 8 quarters, the `schedule` candidate reproduces the
plan's quoted cohort arithmetic table exactly:

| group | plan | measured |
|---|---|---|
| completed inside the ±45d band | 4,899 | **4,899** |
| completed early (before the band) | 5,631 | **5,631** |
| completed late (after the band) | 1,769 | **1,769** |
| never did 20k (skippers + churned) | 1,626 + 952 = 2,578 | **2,578** |
| due total | 14,877 | **14,877** |
| base rate | 70.9% | **70.9%** (10,530/14,877) |
| early % of completers | ~46% | **45.78%** |

**Median lag: exact match, once the reference point is identified.** "Median and IQR of actual -
predicted" was computed two ways for `schedule`:
- per-vehicle predicted date (what the bake-off table below uses for every candidate, since it's
  the only version that varies by candidate): median **-80.0d**, IQR [-183, +28], p10/p90
  [-260, +123].
- **quarter-END as the reference** (same for every vehicle in a cohort quarter, i.e. "due by the
  end of the quarter"): median **-119.0d**, IQR [-224, -9], p10/p90 [-305, +86].

The quarter-end version reproduces the plan's quoted "-119 days" **exactly**. The plan's quoted p10
(-350) and p90 (+143) are in the same range as measured (-305 / +86) but not identical — plausibly a
slightly different service-history snapshot or first-vs-any-occurrence convention in the earlier,
unsaved analysis. Combined with the exact cohort-count match, this is strong confirmation the
harness measures the same mechanism as the earlier analysis.

## Bake-off table (aggregated over 2024Q1..2025Q4, milestone 20k, grace ±45d)

Lag = actual completion date - candidate's per-vehicle predicted date, completers only (early +
in-band + late; excludes never-did-it).

| candidate | due | completers | early % | in-band % | late % | median lag (d) | IQR (d) |
|---|---:|---:|---:|---:|---:|---:|---:|
| schedule | 14,877 | 12,299 | 45.78 | 39.83 | 14.38 | -80.0 | [-183, +28] |
| **calibrated** | 15,808 | 13,285 | **24.40** | 42.74 | 32.86 | +7.0 | [-93, +115] |
| mileage | 7,319 | 6,632 | 30.11 | 58.87 | 11.02 | -34.0 | [-123, +18] |
| interval | 13,828 | 11,048 | 61.07 | 30.49 | 8.44 | -133.0 | [-244, -37] |
| earliest | 15,039 | 12,573 | 34.20 | 49.05 | 16.75 | -25.0 | [-160, +46] |

`never` (skippers + churned, out of the `due` total): schedule 2,578; calibrated 2,523; mileage 687;
interval 2,780; earliest 2,466.

Interval-candidate diagnostic: the population median days-per-PMS-block it falls back to ranged
124-126 days across the 8 cutoffs (n=32k-42k multi-visit pairs) — i.e. real signal, not the
hardcoded 182-day fallback, but as flagged in the definition above almost no vehicle has its OWN
day-interval before a 20k due date (only one prior milestone exists), so the model is really "your
own km-per-block, on the population's typical cadence," not a fully personalized interval.

## Gate: FAILED

**Gate is: no candidate gets the early group under ~20% of completers. All five fail it, and by a
wide margin** — best is `calibrated` at 24.40%, still 4+ points over the line; the other four range
30-61%. `interval` is worse than the current `schedule` baseline (61% vs 46%). `mileage` has real
promise on the metrics that do transfer (58.87% in-band, best of any candidate) but only covers
7,319 of the 14,877 due vehicles (49%) — the rest are excluded for lacking 30+ days of pre-cutoff
history, which itself correlates with being new/without-a-10k, i.e. it silently drops a
non-random slice of the population rather than actually improving coverage.

**No candidate is a viable due-date fix on its own.** Per the plan's explicit instruction, stopping
here — Phases 2-4 (relabel, matching test sets, train/evaluate) are not attempted. `due_date.py` is
left in the repo as the analysis module (five working candidate functions) but is NOT wired into
the `--date-model` CLI flag on either builder, since that wiring exists to serve the relabel this
gate blocks.

## Reading the failure

Even `calibrated` — built from medians measured directly on the population of vehicles that
*completed* the milestone (audit already flags this as "biased early" precisely because it's
fitted on completers) — only pulls the early group down to 24%, not under 20%. That the single
candidate most mechanically tuned to "when do completers actually complete" still leaves a quarter
of completers finishing before their own predicted window suggests the 46% figure is not primarily
a *due-date-formula* problem: the spread in when different vehicles reach a milestone (relative to
first service) is wide enough that no single formula — schedule, mileage, interval, or a calibrated
median — collapses it into a narrow band. Shrinking the early group further would need either a
per-vehicle/per-cohort hazard model (survival estimate, as the plan's "Not in scope" section already
flags as the long-term fix) or accepting a wider grace band, which drifts back toward the "ever
completed" question the plan is trying to move away from.
