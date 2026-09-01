# Feature cutoff audit — 20k, 2026-08-17

Purpose: before relabelling training from service history (which makes training cohorts span years
rather than sitting near the extract date), establish which of the 156 shipped 20k features are safe
under a cohort-specific cutoff and which cannot be.

Two tests were needed, because one alone is misleading.

## Test 1 — move the cutoff, hold the cohort fixed

4,000 fixed VINs (all with an expected 20k date ≤ 2023-12-31) pushed through
`process_service_data()` twice, at `filter_date` 2023-12-31 and 2025-12-31, `is_test=True` so the
same VINs survive both runs. Diff per column.

| source | cutoff-sensitive | static |
|---|---|---|
| EDA-direct | 1 | 14 |
| EDA-onehot | 0 | 24 |
| derived | 21 | 71 |

**Runtime: 55s per run for 4,000 rows.** This is what makes per-quarter cohort builds affordable.

Reading: the 21 cutoff-sensitive derived features are the pipeline working correctly — `serv1` is cut
at `filter_date` ([pmstrainfeatureeng_refactored.py:428](../pmstrainfeatureeng_refactored.py#L428)),
so history-derived features move when the cutoff moves. They are safe **provided the cutoff is set
per cohort**.

**This test says nothing useful about the EDA columns.** Only 1 of 38 moved, not because they are
time-invariant but because the pipeline never re-cuts EDA at all — it is a single 2026 snapshot
merged in wholesale. A current-state field looks "static" here precisely because it is frozen at its
latest value. Hence test 2.

## Test 2 — diff two EDA extracts of the same vehicles

`legacy/EDA - Q3 2025.csv` vs `data/EDA_Q2-2026.csv` (read-only), 126,876 vehicles present in both,
matched on `Vehicle Key`. A column that changes between extracts is recomputed at extract time, so
it describes the vehicle *today*, not at the cohort's due date.

### Unsafe — recomputed per extract, reaching the 20k model

| EDA column | % rows changed | shipped feature columns |
|---|---|---|
| `Purchase Age` | 23.43% | `Purchase Age` |
| `Last Service Mileage` | 11.65% | `Last Service Mileage` |
| `Last PMS Mileage` | 10.09% | `Last PMS Mileage` |
| `Vehicle Service Status` | 5.12% | `..._Active`, `..._InActive`, `..._Lapsed` |
| `Total Survey` | 4.02% | `Total Survey` |
| `Total Promoter` | 3.44% | `Total Promoter` |
| `PMS Status` | 3.08% | `..._Non PMS`, `..._PMS Partial` |
| `Service Contract Status` | 2.05% | `..._Active`, `..._Expired`, `..._No SC Purchased` |
| `Total Passive` | 0.90% | `Total Passive` |
| `Total Detractor` | 0.84% | `Total Detractor` |

**15 shipped feature columns**, listed in `eda_currentstate_features.json`.

### Safe — static between extracts, keep

`CC` (0.02%), `Weight` (0.02%), `Height` (0.02%), `Wheel Base` (0.01%), `Number of Cylinders`
(0.01%), `Franchise_*` (0.04%), `New / Used_*` (0.03%), `AMA or Non AMA_*` (0.00%),
`Warranty Status_Yes` (0.00%), `Gender_*` (0.39%), `First Service Status_*` (0.35%),
`Contract_PurchaseStatus_Final_*` (0.22%), `Tyre Revenue` (0.11%), `Battery Revenue` (0.23%),
`Brake Revenue` (0.48%).

The three revenue fields are cumulative and *ought* to drift, but measurably do not (all < 0.5%),
most likely because they are sparsely populated. Kept on the evidence; worth re-checking if they
ever rank highly in permutation importance.

### Columns not in the 20k feature list but load-bearing elsewhere

| column | % changed | note |
|---|---|---|
| `Vehicle_Key_ExpectedServices` | **49.33%** | first term of the `legacy` `PMS_Delay` |
| `Current Age` | 26.23% | already in `LEAK_COLS` |
| `Service Frequency` | 11.75% | confirms the known finding; not in the 20k list, check other milestones |
| `Vehicle_Key_Actual_Service` | 10.70% | already in `LEAK_COLS` |
| `Last Service - PMS` | 10.20% | the current training label source |
| `Vehicle Service Status` | 5.12% | used as a **cohort filter**, not just a feature |
| `Vehicle Age` | **0.14%** | genuinely static, despite being in `LEAK_COLS` |

`Vehicle_Key_ExpectedServices` changing on half of all rows settles the `PMS_Delay` question:
the `legacy` variant is built on a field that is recomputed every extract. Use `derivedB`
(`2*Years_Since_First_PMS - sum(has_x)`), which is entirely cutoff-bound.

## Caveat on magnitude

These percentages measure ~11 months of drift (Q3-2025 → Q2-2026). Training cohorts under the new
labelling reach back to 2018, so the drift a 2018 cohort would carry is **much larger than the
numbers above**. Treat every figure here as a lower bound.

## Conclusion

- Drop the 15 EDA current-state columns.
- Keep the 23 static EDA columns.
- Keep all 92 derived columns; they are safe once the cutoff is set per cohort.
- Set `PMS_DELAY_VARIANT=derivedB`.
- `Vehicle Service Status` is also a cohort *filter*; it is current-state, so the filter itself is
  mildly anachronistic for old cohorts. Retained (it mostly excludes sold/scrapped vehicles) but
  noted.

Raw output: `feature_cutoff_audit.csv`, `eda_snapshot_diff.csv`, `eda_currentstate_features.json`.
