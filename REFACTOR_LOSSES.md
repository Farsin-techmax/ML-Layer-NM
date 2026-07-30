# What the test-set refactor lost

Analysis of `prepare_test_set.py` + `pmstrainfeatureeng_refactored.py` against legacy
`predservicemil_4.py`. Every claim below has a line reference or a measured number.

---

## 0. First: the column question

**Nothing I wrote removes columns.** The column reduction is in your existing code, at
[prepare_test_set.py:162](prepare_test_set.py#L162):

```python
final_cols = ["VIN", "TargetFlag"] + selected_features     # -> 35 columns
final_test_set = test_features[final_cols]                 # line 170: the cut
```

`process_service_data` returns ~167 columns. Line 170 throws away ~132 of them.

The reason it's wrong is **which** feature list it filters by. There are two different
`selected_features.json` files in this repo and they are not interchangeable:

| File | Count (20k) | Produced by | Meaning |
|---|---|---|---|
| `models/20k/selected_features.json` | **33** | `feature_sel()` MI ≥ 0.85 cumulative, [pmstrainfeatureeng_refactored.py:1035](pmstrainfeatureeng_refactored.py#L1035) | a *feature-engineering* shortlist |
| `models/models_alan/20k/selected_features.json` | **122** | `retrain.py` after the test-schema intersection, [retrain.py:188](retrain.py#L188) | **what the trained model actually consumes** |

`prepare_test_set.py:154` reads the **33** list. Your models need the **122** list.

### Measured consequence

Feed `prepare_test_set.py`'s output to the matching `models_alan` model and this is what happens
(`score_milestone.py` / `retrain.py` zero-fill any missing column):

| Milestone | Model needs | Present in test file | **Zero-filled** | % of inputs dead |
|---|---|---|---|---|
| 20k | 122 | 27 | 95 | **77.9%** |
| 30k | 123 | 31 | 92 | **74.8%** |
| 40k | 126 | 28 | 98 | **77.8%** |
| 50k | 118 | 29 | 89 | **75.4%** |
| 60k | 113 | 29 | 84 | **74.3%** |
| 70k | 124 | 35 | 89 | **71.8%** |
| 80k | 119 | 39 | 80 | **67.2%** |
| 90k | 134 | 36 | 98 | **73.1%** |
| 100k | 136 | 35 | 101 | **74.3%** |

Contrast with the test files you actually use today (`tests_alan/testdataq1_q2/{m}kPMSTestDataQ1.csv`,
~173 columns — the *unfiltered* `process_service_data` output):

| Milestone | Model needs | Present | Zero-filled |
|---|---|---|---|
| 20k … 100k | 113–136 | **all of them** | **0** |

That is the whole point. The working test files are 173 columns because nobody filtered them. The
moment you apply the 33-feature filter, three quarters of the model's inputs become constant zero and
the probabilities are noise.

**So the fix is to remove a filter, not add one** — which is what the migration guide recommended.
The 33-column file is an artifact of pointing at the wrong JSON.

---

## 1. `Next{m}K_Due` changed meaning — highest-impact finding

This column decides which vehicles land in which quarter, and it gates training at
[retrain.py:88-89](retrain.py#L88-L89). Its definition changed completely.

**Legacy** ([predservicemil_4.py:2562-2565](predservicemil_4.py#L2562-L2565)):

```python
filtered_dfnew[f"Next{last_service_code}K_Due"] = filtered_dfnew.apply(
    lambda row: calc_next_due(row["Last Service Date - PMS"], row["predicted_interval_months"]), axis=1)
```

where `predicted_interval_months = Avg_Service_Interval_PMS1 * (target_no - last_service_no)`
([features.py:1318](features.py#L1318)).

Read that: **last actual PMS date + (this vehicle's own learned interval × milestones remaining).**
A per-vehicle behavioural forecast.

**Refactored** ([pmstrainfeatureeng_refactored.py:165](pmstrainfeatureeng_refactored.py#L165), renamed at
[:627](pmstrainfeatureeng_refactored.py#L627)):

```python
missed[f"Expected{svc}Date"] = missed["FirstSrvDate"] + pd.DateOffset(months=months_to_add)
...
filtered_dfnew = filtered_dfnew.rename(columns={f'Expected{last_service_code}kDate': f'Next{last_service_code}K_Due'})
```

`months_to_add = milestone/10 * 6`. Read that: **first service date + a flat 6 months per 10k.**
Identical for every vehicle. No behaviour, no learned interval.

`calculatenxt_service_interval` still exists at [features.py:1314](features.py#L1314) and
`calc_next_due` still exists — **both have zero callers in the refactored pipeline.** Legacy merged
`nextservinterv` into the matrix at [predservicemil_4.py:2464](predservicemil_4.py#L2464); the
refactor does not, so `predicted_interval_months` never reaches the model either.

**Why this matters twice over:**
1. A feature was lost (`predicted_interval_months` — the vehicle's own service cadence).
2. The *cohort definition* changed. A slow-driving vehicle and a fast-driving vehicle with the same
   first-service date now get the same due date, so they fall in the same quarter. Legacy separated
   them. Every quarter-window filter, and the `<= 2025-12-31` training cutoff, now sorts vehicles by
   a calendar formula instead of by predicted behaviour.

`prepare_test_set.py` partially compensates with `--enable-mileage-projection` (lines 32–56), which
projects from mileage rate. But it's **off by default** (line 105 is `store_true`), and it projects
from `Last_Mileage` / days-since-first-service, not from the learned PMS interval. Different quantity.

---

## 2. The non-PMS cohort is gone — 33.6% of the fleet

Legacy built **two** cohorts and concatenated them
([predservicemil_4.py:2716-2719](predservicemil_4.py#L2716-L2719)):

```python
nompmsbase = NonPMS(rfmdf, filterdate, last_service_code, ...)
finalbase = pd.concat([maindf, nompmsbase], axis=0)
```

`NonPMS()` (line 2030) selects on `Vehicle_Key_ExpectedServices == @lscode` (line 2061) and works off
`Description == 'Others'` service history (line 2133) — vehicles that are *due* for the milestone but
whose service record is non-PMS work only.

`prepare_test_set.py:16` opens with:

```python
df = df.query("`Last Service - PMS` != '-'").copy()
```

That excludes exactly those vehicles. Measured on `data/EDA_Q2-2026.csv`:

```
total rows                   130,759
Last Service - PMS == "-"     43,949   (33.6%)  <- silently dropped
Last Service - PMS != "-"     86,810   (66.4%)  <- the only rows kept
```

`Vehicle_Key_ExpectedServices` is still present in the EDA file, and the refactored code touches it
only to print its dtype ([:682](pmstrainfeatureeng_refactored.py#L682)) and drop it
([:807](pmstrainfeatureeng_refactored.py#L807)). It is never used to select a cohort.

**Impact:** a third of the fleet cannot be scored at all. Not "scored badly" — absent. For training
this is arguably defensible (no PMS history → few usable features), but for *prediction* those are
real vehicles a dealership wants a turn-up probability for.

---

## 3. Cluster maps are recomputed instead of reused — this is why you had to drop them

**Legacy treated training clusters as an input**
([predservicemil_4.py:2307-2309](predservicemil_4.py#L2307-L2309)):

```python
modclus = pd.read_csv(f'validatecode/Model_clusters_{last_service_code}.csv')  # (Output from Training)
varclus = pd.read_csv(f'validatecode/Variant_clusters_{last_service_code}.csv')
natclus = pd.read_csv(f'validatecode/Nationality_clusters_{last_service_code}.csv')
```

then assigned prediction rows into those existing clusters via `map_cluster()` (lines 2653-2658).
Cluster "4" at inference meant the same group as cluster "4" in training, by construction.

**Refactored recomputes them from scratch** on whatever cohort it's handed
([pmstrainfeatureeng_refactored.py:795-812](pmstrainfeatureeng_refactored.py#L795-L812)):

```python
pms_imputed, clustersnat = hierarchical_gower_clustering(pms_imputed, mergedfnat, ...)
...
clustersnat.to_csv(f'validatecode/Nationality_clusters_{last_service_code}.csv', index=False)
```

Note it writes **the same filenames legacy read**. The artifact survived; the consumer didn't.
`map_cluster` is still defined at [features.py:1272](features.py#L1272) with **zero callers** in the
refactored pipeline.

**This is the direct cause of the guard at [retrain.py:143](retrain.py#L143):**

```python
_clust = [c for c in X.columns if '_Cluster_' in c]
X = X.drop(columns=_clust)   # "cluster 4 in train is a different group than cluster 4 in test"
```

That comment is correct — but the instability is self-inflicted. Legacy had no such problem, so it
never had to drop the features. Three engineered features (Nationality / Model / Variant cluster) plus
all their one-hot expansions are permanently unavailable to the model, and the fix already exists in
the codebase, unused.

### Related: a silent Gower substitution

[features.py:1462-1502](features.py#L1462-L1502) wraps the `gower` import in `try/except` and falls
back to a hand-written O(n²) pure-Python distance if the package is missing. The fallback does not
reproduce the real package's weighting or NaN handling, so clusters differ depending on whether
`pip install gower` ran. `requirements.txt` doesn't list it (per CLAUDE.md). Nothing warns you.

---

## 4. The mileage ceiling filter is gone

**Legacy** ([predservicemil_4.py:2569-2570](predservicemil_4.py#L2569-L2570)):

```python
threshmil = (last_service_code * 1000) + 10000
fnlupd = fnlupd.query("`Last Service Mileage` < @threshmil")
```

A vehicle predicted for 60k must have last been seen under 70,000 km. Beyond that it has driven past
the milestone and the question is moot.

`threshmil` appears **0 times** in `pmstrainfeatureeng_refactored.py` and **0 times** in
`features.py`. 5 occurrences in legacy.

The refactor has a partially overlapping guard — `prepare_test_set.py:74-80` excludes VINs with a
recorded `Service_Num >= milestone` before the cutoff. But that keys on *service records*, not
mileage. A vehicle at 95,000 km whose 60k service was never recorded passes the refactored filter and
would have been caught by `threshmil`.

---

## 5. The per-stratum cohort loop is gone

Legacy looped over every *prior* milestone
([predservicemil_4.py:2304](predservicemil_4.py#L2304), [:2316-2318](predservicemil_4.py#L2316-L2318)):

```python
service_history = list(range(10, last_service_code, 10))    # target 60k -> [10,20,30,40,50]
for idx, svc in enumerate(service_history):
    months_back = initial + (len(service_history)-idx-1)*6
    cutoff_date = custom_date - pd.DateOffset(months=months_back)
```

Two things came from this:

**(a) A per-stratum recency window.** Vehicles currently at 10k got a wider look-back than vehicles at
50k. The refactor uses one window for everyone.

**(b) Features computed relative to where the vehicle *is*, not where it's *going*.** Legacy passed
both `last_service_code` (the target) and `svc` (current position):

| Function | Legacy call | Refactored call |
|---|---|---|
| `derive_pms_mileage_features` | `(serv1, dfpmsdate, last_service_code, **svc**)` [:2406](predservicemil_4.py#L2406) | `(serv1, pmsdf, last_service_code)` — 4th arg dropped [:400](pmstrainfeatureeng_refactored.py#L400) |
| `get_last_nonpms_mileage` | `(serv1, **svc**, edaM, ...)` [:2402](predservicemil_4.py#L2402) | `(serv1, mastertrain, ...)` — no `svc` [:391](pmstrainfeatureeng_refactored.py#L391) |
| `derive_appointment_show_features` | `(appointdfN, servM, **svc**, ...)` [:2530](predservicemil_4.py#L2530) | `(appointdfN, servM, **last_service_code**, ...)` [:597](pmstrainfeatureeng_refactored.py#L597) |

The `svc` parameter still exists in the signature — [features.py:169](features.py#L169)
`derive_pms_mileage_features(serv1, dfpmsdate, last_service_code, svc=None)` — and its docstring says
`svc: If provided and == 1, skips skipped_blocks logic`. The refactor never provides it, so the
`skipped_blocks` multiplier now always applies, including to first-service vehicles it was meant to
exempt.

---

## 6. New finding — `LastNonPMSMileage` is computed only for `TargetFlag == 0`

Not in CLAUDE.md. Worth its own look.

There are two versions of this function in `features.py`:

```python
# line 1195 — the one the refactored pipeline calls
def get_last_nonpms_mileage(df_nonpms, mastertrain, ...):
    """Last Non-PMS mileage for TargetFlag==0 VINs (training)."""
    vins = mastertrain.query("TargetFlag == 0")['VIN'].unique()          # <-- negatives only
    last_pms_df = mastertrain.query("TargetFlag == 0")[['VIN','Last PMS Mileage']]...

# line 1235 — the legacy prediction version, now uncalled
def get_last_nonpms_mileagepred(df_nonpms, svc, mastertrain, ...):
    vins = mastertrain['VIN'].unique()                                   # <-- everyone
```

`process_service_data` calls the **first** one, at
[:391](pmstrainfeatureeng_refactored.py#L391) and again at [:396](pmstrainfeatureeng_refactored.py#L396).

So positives never receive a measured `LastNonPMSMileage` — they get NaN, and then
`IterativeImputer(Lasso)` fills it. Negatives get a real reading. Two knock-on effects:

- `LastNonPMSMileage` **is** in every `models_alan` feature list (verified for 20k).
- `Last Service Mileage` is derived from it at [retrain.py:95](retrain.py#L95)
  (`max(Last PMS Mileage, LastNonPMSMileage)`) and is **also** in the model's feature list.

If "value was imputed" correlates with "TargetFlag == 1", the model can read the label off the
imputation pattern. Same family of defect as `Service_Num`, though weaker and indirect.

**Honest limit on this finding:** I could not confirm the magnitude from the artifacts.
`finalmerged20kQ1.csv` is written *after* imputation, so the null pattern is already erased — it
shows 0 nulls in both classes. The class means differ (positives 11,617 vs negatives 9,222) but that
is equally consistent with genuine signal. **The code path is confirmed; the impact is not.** To
settle it you would need to inspect the pre-imputation frame, or refit with the column dropped and
compare.

Note this affects *both* the refactored train and test paths identically (`prepare_test_set.py:92`
sets `TargetFlag` before calling `process_service_data`), so it is not a train/serve skew within the
refactor. But in genuine production scoring, where no `TargetFlag` exists, the column cannot be
computed at all.

---

## 7. Things that look lost but aren't

Being fair — I checked these and they're fine:

**`vhcpreparation` service-code window — equivalent.** Legacy passed `last_service_code - 10` with a
`<=` filter; refactored passes `last_service_code` with a `<` filter
([features.py:899](features.py#L899)). Since service codes are multiples of 10, both resolve to "up to
and including milestone − 10". **No leak, no change.** My first read of this was wrong.

**`map_vhc_history` commented out** at
[pmstrainfeatureeng_refactored.py:567](pmstrainfeatureeng_refactored.py#L567). Legacy called it (2522)
and filled 8 VHC columns (2523-2524) — but then **dropped those same 8 columns** at
[:2727](predservicemil_4.py#L2727) and replaced them with `vhcpreparation` output at
[:2729-2730](predservicemil_4.py#L2729-L2730). Dead work in legacy. Removing it is a cleanup.

**Appointment show / no-show files** (`Appoinment - Showed Up.csv`, `No Show VINs.csv`). Read at
legacy 2310-2311, but every consumer is commented out in legacy too (2438-2446, 2466-2467, 2177-2178).
Already abandoned before the refactor. The live path (`derive_appointment_show_features` off the full
appointments file) survives in both.

**`one_hot()` → `pd.get_dummies`.** Legacy used the helper (2666); refactored uses
`pd.get_dummies(pms, columns=cols_to_encode, dtype=int)` at
[:735](pmstrainfeatureeng_refactored.py#L735). Equivalent output.

**Kept and confirmed present in the refactor:** `wearablesBought`, `LowMileageFreqUsers`, `PMS_Delay`,
`calculate_revenue_spend`, `calculate_bodyshop_count`, `get_non_pms_events`, `branch_visit_features`,
`transform_complaint_features`, `compute_service_features`, `adjust_service_intervals`,
`IterativeImputer(Lasso)`, the `missfeatsdrop` list.

---

## 8. Severity ranking

| # | Finding | Severity | Reversible? |
|---|---|---|---|
| 0 | 33-column filter → 67–78% of model inputs zero-filled | **Critical** | Yes — delete the filter at `prepare_test_set.py:170` |
| 1 | `Next{m}K_Due` is a calendar formula, not a learned interval | **Critical** | Yes — `calculatenxt_service_interval` still exists, uncalled |
| 3 | Clusters recomputed → forced to drop all `_Cluster_` features | High | Yes — `map_cluster` still exists, uncalled |
| 2 | Non-PMS cohort absent → 33.6% of fleet unscoreable | High | No — needs the `NonPMS()` path rebuilt |
| 6 | `LastNonPMSMileage` only for `TargetFlag == 0` | High if confirmed | Yes — `get_last_nonpms_mileagepred` still exists |
| 5 | Per-stratum `svc` parameter dropped | Medium | Yes — signatures still accept it |
| 4 | `threshmil` mileage ceiling gone | Medium | Yes — one line |
| 3b | Silent Gower fallback when package missing | Medium | Yes — pin `gower` in requirements |

## The pattern worth noticing

Findings 0, 1, 3, 5 and 6 share a shape: **the legacy capability is still in the codebase, and simply
has no caller.** `calc_next_due`, `calculatenxt_service_interval`, `map_cluster`,
`get_last_nonpms_mileagepred`, and the `svc=None` parameter are all present, all reachable, all
unused. The refactor extracted the functions into `features.py` correctly but wired up only a subset
of them in the new `process_service_data`.

That's the good news: most of this is reconnection, not reconstruction. The exception is finding 2 —
the non-PMS cohort — which needs the `NonPMS()` logic genuinely rebuilt.

## What this changes about the S3 plan

Nothing structural, but the ordering matters. `S3_MIGRATION_GUIDE.md` Stage 6 Step 6.5 is a
**byte-parity** gate — it proves the migration changed nothing. It will pass whether or not these
defects are present, because it compares the refactored pipeline to itself. Fix these before or after
the migration, not during it, or you lose the ability to tell a port bug from a data-logic change.
