# Investigation brief: Nissan PMS turn-up prediction

You are being handed an in-flight ML project to audit and advance. Read this whole brief before
touching anything. It tells you what the project is for, what has already been tried, what keeps
going wrong, and what "done" looks like. It is written to save you rediscovery time — but every
factual claim in it is reproducible from the repo, and if you find one that is wrong, say so and
show the measurement.

Repo root: `c:\Techmax\cwf\code`. All scripts hardcode paths relative to cwd — always run from root.

---

## 1. What the project predicts

A Nissan dealership group wants to know, ahead of each quarter, **which vehicles will turn up for
their scheduled Periodic Maintenance Service (PMS)**. Services happen at mileage milestones —
10k, 20k, 30k … 100k km. The business use is a marketing campaign: the dealership takes the list of
vehicles "expected" for a milestone this quarter and contacts them, so a good model has to produce
both a defensible cohort and a probability score that means something.

Architecture: **one PyTorch binary classifier per milestone**. Two stages.
1. Feature engineering builds a wide numeric per-VIN matrix with a `TargetFlag` label.
2. Per-milestone training/eval on that matrix.

Inputs under `data/` (all gitignored, present on disk): an EDA master sheet (130,759 vehicles,
100 columns), Service History, RFM segments, Appointments, VHC (Vehicle Health Check — dealer
inspections where customers invoice / defer / decline recommended work), and Digital Sessions.

`Service_History_Q2-2026.csv` ends **2026-06-29**. (An earlier revision of `CLAUDE.md` claimed
2026-12-06 — that was an artifact of the day-first parsing bug described in §5(j), now fixed. If you
see 2026-12-06 anywhere, the doc is stale.) Labels are a function of the extract date; always record
which service-history file produced a metric. A separate July 2026 service log exists outside the
base history and is used only by the overlap experiment in §6.

---

## 2. Read these first, in this order

- **`CLAUDE.md`** — the project's own running record. Dense, heavily revised, and the single most
  valuable document here. It is largely accurate and carries measured numbers with dates. **Treat it
  as authoritative but not infallible** — it has been wrong before and says so in places where an
  earlier revision was superseded. Where it disagrees with the code, the code wins; report the drift.
- `pms_model.py` — shared policy layer. `LEAK_COLS`, `DERIVED_FEATURES`, `HAS_X_EXCLUDE`,
  `DROP_RFM`, `MAX_PLAUSIBLE_MILEAGE_KM`, plus `build_inference_matrix()`, `load_artifacts()`,
  `derive_inference_features()`, `add_milestone_flags()`. Policy edits belong here, not in callers.
- `pmstrainfeatureeng_refactored.py` — training-matrix builder. Label modes live here.
- `prepare_test_set.py` — test-cohort builder. Must stay in step with the above.
- `retrain.py` — per-milestone trainer. Read the long comments; several encode hard-won findings.
- `features.py` — ~60 feature-derivation functions, shared by both pipelines.
- `due_date.py` — due-date models (`schedule`, `calibrated`, `mileage`, `interval`, `burn_rate`).
- `REFACTOR_LOSSES.md`, `refactored_test_dir/feature_cutoff_audit.md` — prior audits, line-referenced.

**Docs known to be WRONG** — do not reason from them: `refactored_test_dir/old_vs_refactored_comparison.md`
and `later/refactoring_summary.md` both claim the refactor moved the target to "showed up in the
exact quarter" and that this cost 15–20 accuracy points. That is false as a description of the code.

---

## 3. The two pipelines, and a terminology trap

**Current pipeline** (`features.py`-based): the scripts listed above. This is the live code.

**Legacy**: two things that must never be conflated.
- **`/legacy`** — an archive DIRECTORY holding the earliest script versions, `parked/`, its own
  `venv/`, and older/smaller data snapshots at its root. **It has never been executed.** It contains
  no model artifacts, no feature lists, no metrics, no logs. Nothing reads from it. Any "legacy
  accuracy" quoted from memory or old docs cannot have come from here — `legacy/training_run.py`
  has its test split and every metric line commented out, early-stops on *training* loss, and has no
  validation split at all. **Never attribute anything in `models/` to `/legacy`.**
- **"legacy artifacts in `models/{m}k/`"** — `model.pt`, `feature_list.json`, `scaler.pkl`,
  `training/`. Produced by the older root-level monoliths (`pmstrainfeatureEng_6.py`,
  `predservicemil_4.py`), not by `/legacy`. Always say this in full.

When the project owner says "legacy", they mean the `/legacy` directory.

`predservicemil_4.py` **executes at import** — read it, never import it.

`feature/pytorch-refactor` is a git submodule pinned to a stale nested clone of this same repo.
Grep and glob hit it and return duplicate matches. Never edit there.

---

## 4. What the refactor changed relative to the monoliths

Directionally: deduplication (`BinaryClassifier` was defined 5×, `add_milestone_flags` 3×), a shared
policy module, explicit leak lists, per-quarter cohort construction, and CLI-selectable label/date
models. But several changes were **regressions that took months to find**, and this pattern is the
single most important thing to understand about this codebase:

- `PMS_Delay` — legacy computed `Vehicle_Key_ExpectedServices − Vehicle_Key_Actual_Service`, a real
  per-vehicle "how far behind schedule" signal. The refactor replaced the first term with the
  **constant** `milestone/10`, making the feature correlate **−1.0000** with a column that
  `LEAK_COLS` explicitly drops — reintroducing the leak through the back door. Fixed 2026-08-13.
- `has_x` milestone flags — built correctly from cutoff-bounded service history, then
  **overwritten** by `add_milestone_flags()` using post-outcome `Service_Num`. The good version never
  reached a model. `has_50`'s positive-class rate went 0.732 → 1.000. Fixed 2026-08-16.
- `LowMileageFreqUsers` — carried `& (TargetFlag == 1)` inside its own definition, so the flag could
  only ever fire on a positive. `/legacy` had already fixed this; the refactor reintroduced it.
- Blanket train/test column intersection deleted legitimate per-category dummies from *training*.
  Fixed via family-matching; recovered 36 of 41 columns on 20k (129 → 165 features).

**The meta-lesson: this codebase's failures are silent.** Nothing crashes. A leak, a constant
feature, a zero-filled column, or an outcome-selected test set all produce clean-looking runs with
plausible metrics. Assume any number you did not personally reproduce is suspect.

---

## 5. The recurring failure modes

These are classes, not incidents. Each has recurred multiple times in different disguises. When you
find something new, check whether it is one of these wearing a new hat.

**(a) Label leakage.** `Service_Num == milestone ⟺ TargetFlag == 1`, exactly, zero exceptions —
because `Service_Num` is only stamped on turn-up. Everything derived from it inherits this.
Confirmed leaks found so far: `Service_Num`, `Vehicle_Key_Actual_Service`, `PMS_Delay` (pre-fix),
`Service Frequency`, `has_x` (pre-fix), `LowMileageFreqUsers` (pre-fix), `PMS Status_*`.
**Diagnostic signature: single-feature AUC on TEST substantially exceeds AUC on TRAIN.** A genuine
feature does not get stronger on held-out data.

**(b) Labelling convention drives everything.** Four modes exist: `ever`, `window`, `history`,
`candidates`.
- `ever` (long-time production default): positive = completed the milestone *at any point*;
  negative = still pending past its expected date. **No date condition on positives.** This draws
  positives from all time and negatives from one due-date window, so the two classes differ
  systematically in service-history depth — and every history-derived feature inherits that
  difference with the **wrong sign**. On 20k this was catastrophic and measurable:

  | feature | `ever` train AUC | test AUC | verdict |
  |---|---|---|---|
  | `has_10` | 0.3602 | 0.6191 | inverted |
  | `PMS_Count_Prior` | 0.3792 | 0.6002 | inverted |
  | `Years_Since_First_PMS` | 0.4099 | 0.5582 | inverted |
  | `PMS_Freq_PerYear` | 0.3815 | 0.5995 | inverted |

  The model learned "has done more prior services → will NOT turn up". That is backwards, and it is
  a sampling artifact, not behaviour.
- `candidates` (current best): due date = earliest of the 12-month schedule and a service-1 →
  service-10 burn-rate projection to the target mileage; requires a genuine `Invoice date` (sale
  date); labels from raw service history. Both classes drawn the same way. **This fixes the
  inversion** — all four features above move to 0.58–0.63 on train, matching test.
- `window` was tried and **lost AUC nearly everywhere** (40k −0.133, 60k −0.027) while improving
  calibration. Do not re-adopt it without addressing why.

**(c) Cohort construction / due-date modelling.** Who is "expected this quarter" is a modelling
choice with large consequences. Findings on record: the mileage projection is **worse** than the
time-based date (leave `--enable-mileage-projection` off); `--date-model calibrated` **lost** 6 of 8
comparisons and its calibration was fitted on completers only (survivorship bias — only 38.7% of
60k-eligible vehicles ever complete, so a median over completers answers the wrong question; needs
Kaplan-Meier, not a median).

**(d) Train/test schema mismatch.** Missing columns are silently zero-filled at inference. This was
historically the dominant cause of bad metrics (~52% of inputs zero-filled at one point). Guarded
now via `_test_feature_cols` reconcile + `split_missing()` reporting `ZeroFill` (real gaps — must be
0) vs `DummyFill` (absent categories — harmless).

**(e) EDA current-state contamination.** ~15 EDA columns are recomputed per extract rather than
being static attributes, so a 2016 cohort would otherwise carry 2026 values. Listed in
`refactored_test_dir/eda_currentstate_features.json`. `Service Frequency` is the worst offender
(single-feature AUC 0.9449 test vs 0.8300 train — textbook signature (a)).

**(f) Data quality.** `LastNonPMSMileage` contains 73 rows between 1.0M and 9.95M km — physically
impossible. `IsolationForest(contamination=0.005)` does **not** catch them (0 of 73 removed). At
`RobustScaler` output they reach magnitude 510 vs a normal ±3. Now winsorized at a 500,000 km domain
cap inside `derive_inference_features()` (the shared train+score path). Also: `Invoice date` parses
for only 60,411/130,759 EDA rows (46%), and 13,950 vehicles have EDA saying "no PMS" while raw
service history shows a PMS row — **the two sources disagree and both builders trust EDA**.

**(g) Evaluation integrity — the most dangerous class.** Two separate incidents:
- `eval_test_metrics.py` was reading `tests_alan/testdataq1_q2/*.csv` while `prepare_test_set.py`
  wrote to `test_sets/`. The former had 44 (Q1) / 11 (Q2) model inputs entirely absent and hard-zeroed
  for **100% of rows**, and reported *better* numbers because the zeroed columns were exactly the
  inverted family from (b) — accidental ablation.
- Those `tests_alan` files are **not a forward-looking cohort at all**. 1,784/1,790 (Q1) and
  1,778/1,778 (Q2 — literally 100.0%) of their positives completed inside the target quarter, which
  is impossible for an honestly-constructed cohort. Membership rule is
  `Vehicle_Key_Actual_Service ≤ 2` with `TargetFlag = (Service_Num == 20)` — an outcome-selected
  answer key, not a due-date cohort. Any metric measured on them is void.

**(h) Threshold / calibration drift.** AUC and accuracy diverge badly here. Example: `40k_cand_wins_2021`
has AUC 0.912 but accuracy 69.2%. `models/{m}k/threshold.json` is used by `score_milestone.py` but
NOT by `retrain.py`/`eval_test_metrics.py` (which use 0.5), so the same model reports different
numbers depending on the entry point. Most `threshold.json` files are stale.

**(j) Silent data-format bugs — fixed 2026-08-31, and the clearest example of the whole pattern.**
Two bugs that mislabelled cohorts without raising anything:
- **Date convention.** EDA, Service History and VHC are day-first (`29-06-2026`); Appointments is
  ISO. Several call sites parsed day-first files without `dayfirst=True`, so every value whose day
  part was 1–12 silently became a different real date — **33.2% of service records, worst drift 324
  days**. Forcing `dayfirst=True` everywhere is *not* the fix: on pandas 3.0.3 it corrupts ISO values
  carrying a time component (`2026-08-09` → `2026-09-08`). `date_utils.parse_dates()` now inspects
  values and chooses per column; every date-reading site goes through it.
- **Service number.** `Description` is a mileage band, and the same 20k service is spelled three ways
  across files: `11-20` (clean), `Nov-20` (Excel damage), and `11_20` (a manual repair present only
  in Q1 2026, 1,967 rows). `extract_kk` read the third as **1120**, because Python's `int()` accepts
  `_` as a digit separator (PEP 515). `features.resolve_service_num()` now treats the integer
  `Service_Code` as source of truth with `Description` as fallback.

Isolated effect on the 20k Q1 2026 cohort:

  | | rows | positives | rate |
  |---|---|---|---|
  | original (both bugs) | 951 | 270 | 28.4% |
  | dates fixed only | 999 | 277 | 27.7% |
  | Service_Num fixed only | 951 | 751 | 79.0% |
  | **both fixed** | **999** | **821** | **82.2%** |

**No commit caused this regression — a data file did.** The service-history file was replaced on
2026-08-19, which introduced the `11_20` spelling and broke labelling. This is why provenance
matters more than git history here, and why every metric needs its input files recorded.

**(i) Early completers.** In `candidates` mode, vehicles that completed the milestone *before* the
quarter opened are kept as positives (a deliberate business decision — the client wants the full
expected cohort). They are 24–28% of the cohort and 100% positive, so their outcome is already
settled at prediction time and they inflate every headline metric. An `EarlyCompleter` column flags
them. **Always report the not-early slice beside the combined number.**

---

## 6. Current status (verify before trusting)

### Branch state — read carefully, it is not what it looks like

- `origin/main` is at `228d336`, a merge of PR #1 (the 20k candidates + winsorization work).
- Working branch `fix/date-and-service-num-parsing` is at `2e4460b`: **2 commits ahead of main, 1
  behind**. The date/`Service_Num` fixes in §5(j) and the July overlap experiment are on this branch
  and are **NOT merged to main**.

### ⚠ Every model metric below was measured BEFORE the §5(j) parsing fixes

This is the single most important caveat in this brief. The fixes landed 2026-08-31. Artifact dates:

| artifact | dated | status |
|---|---|---|
| `test_20k_Q1_2026_cand.csv` | 2026-09-01 | rebuilt post-fix |
| `test_30k/40k_*_cand.csv` | 2026-08-20 | **stale — pre-fix** |
| `models/20k/` (promoted) | 2026-08-19 | **stale — pre-fix** |
| `models/30k_cand_wins_*`, `models/40k_cand_wins_*` | 2026-08-20 | **stale — pre-fix** |

So the models were trained and scored against cohorts whose labels the parsing bugs had corrupted.
On the `ever`-mode 20k Q1 cohort the fixes moved the positive rate from 28.4% to 82.2% — that is not
a rounding difference, it is a different problem. **Treat the table below as a historical record of
what was measured, not as current truth. Regenerating cohorts and retraining on fixed data is
prerequisite to any new claim.**

Production (`models/{m}k/`) vs candidates+winsorized, threshold 0.5, ROC-AUC Q1/Q2, **all pre-fix**:

| milestone | production | candidates best | best window |
|---|---|---|---|
| 20k | 0.555 / 0.415 | 0.925 / 0.950 | 2021+ |
| 30k | 0.940 / 0.929 | 0.963 / 0.969 | 2023+ |
| 40k | 0.917 / 0.871 | 0.922 / 0.931 | 2023+ |
| 50k–100k | untouched by this work | — | — |

20k was promoted into `models/20k/` (previous artifacts backed up to
`models/20k_ever_backup_20260819/`). 30k/40k candidates models exist but are **not** promoted.

### The cohort itself does not work — probably the most important finding here

`july_candidates_overlap.py` asks the question the client actually cares about: *if we list the
vehicles due for a milestone in July, how many turn up?* It uses a July 2026 service log outside the
base history, with the burn-rate projection pinned at `--cutoff 2026-06-30` so nothing inside the
window informs the prediction. No model, no probabilities — cohort membership against ground truth.

Result: **4,246 candidates, 3,118 actual turn-ups, 274 overlapping — 8.8% recall, 6.5% precision.**
(Measured pre-fix; will move on a rerun, but not by an order of magnitude.)

Two independent causes, both diagnosed in `july_overlap_diagnose.py`:
- A **reachability ceiling of 81–92%** — the invoice-date guard alone drops 7–10% of the vehicles
  that actually turned up, before any date is computed.
- The due date is **unbiased but far too imprecise** for a one-month window.

This matters more than any AUC number in this brief. A model can rank beautifully inside a cohort
that is itself almost disjoint from reality. Goal (2) in §7 — a defensible cohort for the client — is
currently **not met**, and no amount of classifier tuning fixes it.

20k audit results, for calibration on what "verified" means here: metrics reproduce from disk at
delta 0.00e+00 through the public scoring path; artifacts byte-match; seed-stable across 42/7/123/2024
(Q1 AUC 0.926 ±0.008); no `LEAK_COLS` in the feature list; zero schema gaps. Honest not-early
figures are Q1 0.898 / Q2 0.935 (vs 0.925/0.950 combined).

**Open items:**
- **Everything needs regenerating on fixed data** (see the warning above). Cohorts, matrices, models,
  metrics. Until that is done, no number in this repo is trustworthy.
- **Cohort recall is 8.8%.** See above. This is the headline problem.
- The date/`Service_Num` fixes are unmerged on a side branch while `main` carries the 20k promotion
  that was built on corrupted labels. Those two facts are in tension and need resolving.
- `retrain.py`'s `LEAKY_FAMILY_PREFIXES` guard only runs on train-only columns, so a `PMS Status_*`
  column present in **both** train and test bypasses it (this is how `PMS Status_PMS only` reached
  the promoted 20k model). Impact measured as negligible (test AUC 0.5033, fires on 0.66% of rows) but
  the guard is broken.
- Best training window differs per milestone (20k→2021+, 30k/40k→2023+) with no explanation. Could be
  real distribution shift or could be noise. Unresolved.
- 40k calibration is poor at the 2021+ window despite good AUC.
- 50k–100k have not been revisited under `candidates` labelling at all.
- **Artifact sprawl**: ~90 model directories, ~90 feature-list JSONs, 33 training matrices, 29 test
  sets. Most are dead ablation runs. Nothing records which are meaningful.

---

## 7. What the project owner is trying to achieve

Three goals, in priority order. Judge every proposal against them.

1. **Raise evaluation metrics without leaking future data.** Both halves are binding. A metric gain
   that comes from leakage is a regression, not progress — this project has repeatedly produced
   impressive-looking numbers that turned out to be leaks. Every claimed improvement needs a leak
   check attached.
2. **Deliver a presentable expected cohort to the client with a reliable probability score.**
   "Reliable" means calibrated, not just well-ranked. AUC alone is insufficient — the dealership acts
   on the probability. Brier score and calibration curves matter as much as discrimination. The
   cohort itself must also be defensible: the client is told "these vehicles are due this quarter",
   so the due-date model has to be justifiable in business terms, not just statistically convenient.
3. **Clean the whole table.** Feature hygiene end to end — remove dead/constant/leaky columns,
   resolve the artifact sprawl, make the pipeline's state legible.

---

## 8. Your task

Audit the repository and produce a findings report. Specifically:

0. **Re-establish a trustworthy baseline first.** Every model metric in §6 predates the §5(j)
   parsing fixes. Regenerate cohorts and matrices on fixed data, retrain, and re-measure before
   making any new claim. Expect the numbers to move; report honestly if the improvements shrink.
1. **Then verify or refute the §6 numbers.** Do not take them on faith. If they hold post-fix, say
   so with your reproduction; if not, show exactly where they break.
1b. **Attack the 8.8% cohort recall.** This is the highest-value open problem and the one that
   blocks goal (2). The invoice-date guard costs 7–10% of real turn-ups on its own — is that
   trade still worth it? Can the due-date estimate be made precise enough for a monthly window, or
   should the product be a ranked list over a wider window rather than a month-boxed cohort? A
   survival/hazard formulation (Kaplan-Meier or similar) has been suggested but never built.
2. **Hunt for remaining leaks**, especially in the 30k/40k/50k+ paths that have had less scrutiny
   than 20k. Use the AUC_test > AUC_train signature, and check features individually *and in groups*
   (permutation importance under-credits redundant features here — the repo has a documented case
   where `PMS_Delay` measured +0.0005 alone despite a 0.90 single-feature test AUC, because
   `has_x`/`PMS_Count_Prior`/`PMS Status_*` covered for it. Grouped permutation is the valid test).
3. **Assess whether `candidates` labelling is the right final answer** or a local optimum. It fixed
   the 20k inversion, but it keeps early completers as positives, which inflates metrics and may not
   be what a client-facing probability should represent. Is there a formulation that satisfies both
   the business framing and statistical honesty?
4. **Resolve the per-milestone training-window question** (2021+ vs 2023+ vs full span) with evidence
   rather than per-milestone tuning, which risks overfitting to two quarters of test data.
5. **Address calibration**, not just ranking — the client consumes probabilities.
6. **Propose the cleanup** — which of the ~90 model dirs and 33 matrices matter, what the canonical
   pipeline state should be, and how to keep it legible.
7. **Produce a rollout plan covering all nine milestones (20k–100k).** Only 20k/30k/40k have been
   touched; 50k–100k still run `ever`-mode labelling and pre-fix data. The plan must state: the
   order milestones are done in and why; which fixes are global (apply once in `pms_model.py` /
   `features.py`) versus per-milestone (training window, `HAS_X_EXCLUDE`, `DROP_RFM`); what is
   expected to differ at high milestones (thinner cohorts, lower completion rates — only 38.7% of
   60k-eligible vehicles ever complete, so base rates and censoring get worse as milestone rises);
   a single reproducible command sequence per milestone; and a go/no-go gate per milestone so a
   failing one is not promoted just because the batch ran. Do not tune per milestone against Q1/Q2
   2026 — that is 9 chances to overfit two quarters.

---

## 9. Standards of evidence

This project has been repeatedly misled by plausible-sounding claims that nobody measured. Hold to
these:

- **Measure, don't assume.** Every claim gets a number and the command that produced it.
- **Only two quarters of test data exist** (Q1/Q2 2026). It is very easy to overfit decisions to
  them. Prefer explanations that generalise; treat a change that helps one quarter and hurts the
  other as unexplained, not as a win.
- **Distinguish "I verified this" from "the docs say this."** State which.
- **A metric with no leak check is not a result.**
- **Report the not-early slice** alongside any combined number in `candidates` mode.
- **Reproduce before promoting.** Metrics written by the training script are not independent
  verification; re-score from disk through `load_artifacts()` / `build_inference_matrix()` /
  `predict_proba()`.
- **Do not modify `/legacy`.** Read-only archive.
- **If you disagree with a decision recorded in `CLAUDE.md`, say so directly and show the
  measurement.** Several entries there are explicitly marked as superseding an earlier wrong
  conclusion — that revision process is how this project makes progress, and it depends on people
  contradicting the record when the evidence warrants.

- **Record input file provenance with every metric.** §5(j) exists because a data file was swapped
  and nothing tracked it. Git history did not catch it and will not catch the next one.

Environment: in-repo venv at `.\venv\Scripts\python`. No lint config, no build, no pytest.

Two hand-rolled test scripts now exist and are the closest thing to a regression suite —
`test_dayfirst.py` (41 assertions) and `test_service_num.py` (31). They are plain scripts that exit
non-zero on failure: run them with `.\venv\Scripts\python test_dayfirst.py`. **Run both before and
after any change touching date parsing or service-number resolution.** Broader verification means
running the eval scripts and comparing metrics; `eval_test_metrics.py` is the quickest broad check
(all 9 milestones, Q1+Q2, ~2 min).
