# Microsoft Fabric Migration — Beginner Guide

Same shape as `S3_MIGRATION_GUIDE.md`: work through it **one stage at a time**, every stage ends with
a checkpoint and a safe stop point. Nothing in a later stage breaks if you stop after an earlier one.

Estimated: 8 stages, 30–90 min each.

> Read `AWS_VS_FABRIC.md` (bottom of this file) **before Stage 1**. Stage 1 costs money; the rest
> of the decision is reversible, that one is not.

---

## Progress tracker

| Stage | What you build | Done? |
|---|---|---|
| 0 | Understand the picture (no typing) | ☐ |
| 1 | Workspace + capacity + Lakehouse | ☐ |
| 2 | Upload the 7 raw CSVs to OneLake | ☐ |
| 3 | The Environment (torch, gower, kneed…) + your code as a library | ☐ |
| 4 | `pms_paths.py` — the one code change | ☐ |
| 5 | Feature-engineering notebook + the parity test | ☐ |
| 6 | Training notebook + MLflow | ☐ |
| 7 | The pipeline — 9 milestones, one click | ☐ |
| 8 | Scoring → Delta table → Power BI, and pausing | ☐ |

---

# Stage 0 — Understand the picture

**No commands. Just read. 10 minutes.**

## The one idea that clears up most confusion

The S3 guide spends its entire Stage 0 on this problem:

> `os.path.exists("s3://bucket/data/file.csv")` returns **False**. Your script exits before doing
> anything. There are 4 places in your code that do this and they will all fail.

**That problem does not exist in Fabric.** When a notebook has a Lakehouse attached, OneLake is
mounted at a real POSIX path:

```python
import os, pandas as pd
os.path.exists("/lakehouse/default/Files/raw/2026-q2/EDA_Q2-2026.csv")   # -> True
df = pd.read_csv("/lakehouse/default/Files/raw/2026-q2/EDA_Q2-2026.csv")  # just works
```

That is an ordinary filesystem path. `os.path.exists` works. `os.makedirs` works. `open()` works.
`pd.read_csv` works.

**So the whole `s3io.py` download/upload layer — Stage 4 of the S3 guide, ~10 functions — is not
needed here.** You delete a design problem instead of solving it.

### What you still have to change

One thing, and it is the same thing either way: **your paths are relative to the repo root.**

`CLAUDE.md` says *"All data paths are hardcoded relative to cwd — always run from repo root."* A
notebook's cwd is not your repo. So every `data/EDA_Q2-2026.csv` has to become
`{BASE}/data/EDA_Q2-2026.csv` where `BASE` is set once. That is Stage 4, and it is about 15 lines.

The second change is smaller: `retrain.py` reads `sys.argv[1]`, and notebooks have no `argv`. That
becomes a parameter cell.

## What maps to what

| Today | In Fabric |
|---|---|
| `data/*.csv` on your disk | Lakehouse **Files** section → `Files/raw/2026-q2/` |
| `refactored_test_dir/final_processed_{m}k.csv` | `Files/derived/{snapshot}/` |
| `models/{m}k/*.pt, *.joblib` | `Files/models/{m}k/` **+ MLflow model registry** |
| `models/selected_features_{m}k.json` | `Files/models/` (keep it a plain file — one list per model) |
| `metrics_*.json`, `retrained_test_metrics_final.csv` | **MLflow experiments** + a Delta table |
| `predictions/{m}k/scored_*.csv` | Delta table `predictions` → Power BI |
| `python retrain.py 20` ×9, by hand | **Data pipeline**, ForEach over milestones |
| `$env:PMS_SEED`, `$env:PMS_DELAY_VARIANT` | Notebook **parameters**, set by the pipeline |
| `debug.txt`, `pms_processing.log` | `Files/run_logs/{run_id}/` |
| `venv\Scripts\python` | A Fabric **Environment** (Stage 3) |

## Three things about this specific project

**1. Stay on Python notebooks. Do not use Spark.**

Fabric offers both. Your data is 130,760 vehicles and a 261 MB snapshot — it fits in memory on one
machine. A Python notebook runs single-node on 2 vCores and is what your pandas code already expects.

There is a correctness reason too, not just cost. `retrain.py:41` records that two runs on an
*identical* matrix differed by **6.7 accuracy points** when unseeded. Spark adds a second source of
nondeterminism — partition order — on top of that. You do not want to debug both at once.

**2. Your data is personal data.** VINs, customer nationalities, RFM segments. The capacity's region
is chosen once and is painful to change. The S3 guide picked `me-central-1` (UAE); the Fabric
equivalent is **UAE North**. Decide this in Stage 1, deliberately, and write it down.

**3. Upload 261 MB, not 933 MB.** Your `data/` folder is 933 MB but most of it is duplicates —
`.xlsx` twins of the CSVs (~199 MB) and legacy-named copies (`Appoinments2025.csv` is byte-identical
to `Appoinments - Q3 - 2025.csv`; note `Appointments` with a **t** is the new one). Stage 2 uploads
the 7 new-snapshot CSVs only.

### ☑ Checkpoint 0

You can explain, out loud:
- Why Fabric doesn't need `s3io.py` → *OneLake mounts as a real path, so `os.path.exists` works*
- What the one real code change is → *base-path indirection, because notebook cwd ≠ repo root*
- Why not Spark → *data fits in memory, and partition order adds nondeterminism to an already
  seed-sensitive model*

**Safe to stop here.**

---

# Stage 1 — Workspace, capacity, Lakehouse

**Goal:** a workspace on a capacity you can pause, with one empty Lakehouse.

⚠️ **This is the stage that starts billing.** Everything before it was free.

## Step 1.1 — Check whether your organisation already has capacity

Do this **first**. It changes the economics by two orders of magnitude.

Go to **app.fabric.microsoft.com → Settings (gear) → Admin portal → Capacity settings**.

- **If you see an existing F64 (or larger)** — used for Power BI elsewhere in the business — ask to
  be given a workspace on it. Your pipeline is a rounding error of CU on a capacity that is already
  paid for. Marginal cost ≈ **$0**. Fabric wins outright; skip the whole comparison.
- **If you see nothing** — you are buying capacity for this project alone. Read `AWS_VS_FABRIC.md`
  below before continuing.

## Step 1.2 — Start the 60-day trial, not a paid SKU

Fabric offers a free trial capacity (F64-equivalent, 60 days). **Use it for Stages 1–8.** You will
know whether this project belongs on Fabric long before you have to pay for it.

**Settings → Account manager → Start trial.**

## Step 1.3 — Create the workspace

**Workspaces → New workspace.**

- Name: `pms-turnup`
- Advanced → **License mode**: Trial (or your F-SKU)
- **Region**: pick deliberately — see Stage 0 note 2. It cannot be changed later.

## Step 1.4 — Create the Lakehouse

Inside the workspace: **New item → Lakehouse**, name it `pms`.

You get two sections:

```
Files/    Ordinary files. CSVs, .pt, .joblib, .json, logs.
          This is where almost everything in this project lives.

Tables/   Delta tables. Queryable with SQL, readable by Power BI without import.
          Stage 8 uses this for predictions. Ignore it until then.
```

**Your model artifacts go in `Files/`, not `Tables/`.** `best_model.pt` and `scaler.joblib` are
binaries, not tables. A common beginner mistake is trying to force everything into `Tables/`.

## Step 1.5 — Prove it works

**New item → Notebook.** On the left, **Add data items → Existing Lakehouse → `pms`**.

Run this in the first cell:

```python
import os, sys, platform
print("python :", sys.version.split()[0])
print("cwd    :", os.getcwd())
print("mount  :", os.path.exists("/lakehouse/default/Files"))
print("files  :", os.listdir("/lakehouse/default/Files"))
```

**You should see:** `mount : True` and `files : []` (empty — you haven't uploaded anything).

**If you see `mount : False`** → the Lakehouse isn't attached. Look at the left panel; there must be
a Lakehouse listed with a filled radio button marking it *default*.

### ☑ Checkpoint 1

- [ ] You know whether your org already owns capacity (Step 1.1) — **write the answer down**
- [ ] Workspace exists, region recorded
- [ ] Lakehouse `pms` exists
- [ ] `os.path.exists("/lakehouse/default/Files")` is `True`

**Safe to stop here.** If on trial, nothing is being charged.

---

# Stage 2 — Upload the raw files

**Goal:** 7 CSVs in `Files/raw/2026-q2/`, untouched forever after.

## Step 2.1 — The layout

Mirror the S3 guide's layers — the reasoning was right and doesn't change:

```
Files/raw/2026-q2/       Originals. NEVER changed, NEVER overwritten. Your safety net.
Files/staged/2026-q2/    Parsed once, saved as parquet.
Files/derived/2026-q2/   The feature matrices.
Files/models/2026-q2/    Trained artifacts.
Files/run_logs/{run_id}/ Logs + manifest.
Files/code/              features.py, pms_model.py, milestone_features.py  (Stage 3)
```

## Step 2.2 — Upload

Small files: Lakehouse → **Files → … → Upload → Upload files** in the browser. Fine for RFM (1.4 MB).

For the 118 MB service history the browser upload is slow and drops. Use **OneLake File Explorer**
instead — it syncs OneLake to Windows Explorer like OneDrive:

1. Install from https://www.microsoft.com/download/details.aspx?id=105222
2. Sign in; a `OneLake - Microsoft` folder appears in Explorer
3. Copy the 7 files into `OneLake - Microsoft\pms-turnup\pms.Lakehouse\Files\raw\2026-q2\`

Upload exactly these:

| File | Size |
|---|---|
| `Service_History_Q2-2026.csv` | 118.19 MB |
| `EDA_Q2-2026.csv` | 49.58 MB |
| `Appointments_Q2-2026.csv` | 46.69 MB |
| `VHC_Q2-2026.csv` | 29.84 MB |
| `Digital_Sessions_Q2-2026.csv` | 2.46 MB |
| `RFM_Q2-2026.csv` | 1.36 MB |
| `Service Code Desc.csv` | 0.01 MB |

**Do not upload** the `.xlsx` files, the legacy-named duplicates, or `feature/pytorch-refactor/` —
that last one is a git submodule pinned at `dd31462`, a stale nested clone of this same repo. It
would double everything.

## Step 2.3 — Verify

```python
import os
p = "/lakehouse/default/Files/raw/2026-q2"
for f in sorted(os.listdir(p)):
    print(f"{os.path.getsize(os.path.join(p,f))/1024**2:9.2f} MB  {f}")
```

**Pass condition:** 7 files, sizes matching the table above to 2 decimals. A short file is a
truncated upload — delete and re-copy it.

### ☑ Checkpoint 2

- [ ] 7 files under `Files/raw/2026-q2/`, all sizes match
- [ ] No `.xlsx`, no legacy duplicates, no `feature/pytorch-refactor`

**Safe to stop here — and it's a genuinely useful stop.** Your raw data now has an off-machine,
redundant copy.

---

# Stage 3 — The Environment and your code

**Goal:** a notebook that can `import features` and `import torch`.

## Step 3.1 — What's missing from the default runtime

Fabric's runtime ships pandas, numpy, scikit-learn, matplotlib, seaborn. It does **not** reliably
ship what this project needs on top:

```
torch  gower  kneed  fastparquet  duckdb  shap  openpyxl  joblib
```

`gower` and `kneed` are the ones that will definitely be absent — they're used by the Gower
clustering in `features.py`.

## Step 3.2 — Create an Environment

**New item → Environment**, name `pms-env`.

**Public libraries → Add from PyPI**, and pin the versions from your `requirements.txt` so results
stay reproducible:

```
torch==2.13.0
scikit-learn==1.9.0
pandas==3.0.3
numpy==2.4.6
gower==0.1.2
kneed==0.8.6
fastparquet==2026.5.0
duckdb==1.5.4
shap==0.52.0
joblib==1.5.3
openpyxl==3.1.5
```

Click **Publish**. This takes 10–20 minutes the first time. It is a one-off.

> **Cold start:** a notebook on a custom Environment takes ~3–5 min to start versus ~10 s on the
> default. That is per session, not per notebook run within a session. Annoying interactively,
> irrelevant for the scheduled pipeline.

> **Version pinning matters more here than usual.** `torch==2.13.0` is not cosmetic — the seeding in
> `retrain.py:44-45` (`np.random.seed`, `torch.manual_seed`) only reproduces within a version.

## Step 3.3 — Upload your shared modules

Three files are imported, not run: `features.py`, `pms_model.py`, `milestone_features.py`.

Copy them into `Files/code/` (OneLake File Explorer again), then at the top of every notebook:

```python
import sys
sys.path.insert(0, "/lakehouse/default/Files/code")
```

> **Why `Files/code/` and not a .whl in the Environment?** Because you will edit `pms_model.py`
> during the migration, and re-publishing an Environment takes 15 minutes each time. Once things
> settle, packaging them as a wheel is tidier. Not now.

## Step 3.4 — Prove it

New notebook, attach `pms-env` (**Environment** dropdown in the toolbar) and the `pms` Lakehouse:

```python
import sys
sys.path.insert(0, "/lakehouse/default/Files/code")

import torch, gower, kneed, sklearn, pandas as pd
from features import *
from pms_model import BinaryClassifier, LEAK_COLS, DERIVED_FEATURES

print("torch  :", torch.__version__)
print("cuda   :", torch.cuda.is_available())   # False is fine and expected
print("model  :", BinaryClassifier(input_dim=165))
print("leaks  :", LEAK_COLS)
```

**You should see:** the version, `cuda : False`, the 64→32→1 network, and the leak column list.

`cuda : False` is correct. `retrain.py:281` falls back to CPU, and a 165-feature MLP over ~8,000
rows trains in seconds on CPU. **Do not pay for a GPU SKU for this model.**

### ☑ Checkpoint 3

- [ ] `pms-env` published
- [ ] `import torch`, `import gower`, `import kneed` all succeed
- [ ] `from features import *` succeeds
- [ ] `from pms_model import BinaryClassifier` succeeds

**Safe to stop here.** No project file has been modified yet.

---

# Stage 4 — `pms_paths.py`, the one code change

**Goal:** one small module so the same code runs on your laptop *and* in Fabric, unchanged.

This is Fabric's equivalent of `s3io.py` — except it is ~15 lines instead of ~10 functions, because
there is no download/upload to manage.

## Step 4.1 — What it does

`pms_model.py` already has `resolve_path()`, which falls back to `tests_alan/<path>` when a file is
missing at root. Extend that idea with one base directory:

```python
# pms_paths.py
import os

# Fabric mounts the default Lakehouse here. On a laptop this path doesn't exist,
# so BASE falls back to the repo root and everything behaves exactly as it does today.
_FABRIC = "/lakehouse/default/Files"
BASE = os.environ.get("PMS_BASE") or (_FABRIC if os.path.isdir(_FABRIC) else ".")

def p(*parts):
    """Resolve a repo-relative path against BASE. Creates nothing."""
    return os.path.join(BASE, *parts)

def ensure(*parts):
    """Same, but create the directory first. Use for outputs."""
    d = os.path.join(BASE, *parts)
    os.makedirs(d, exist_ok=True)
    return d
```

The auto-detect is the point: **you change no code paths when moving between laptop and Fabric.**

## Step 4.2 — Wire it into `resolve_path()`

Rather than editing every script, change the one function they all already call. In `pms_model.py`,
make `resolve_path()` try `p(path)` before the existing `tests_alan/` fallback.

Then the hardcoded literals become `p(...)` calls in four places:

| File | What to wrap |
|---|---|
| `pmstrainfeatureeng_refactored.py` ~944–950 | the 7 input paths |
| `pmstrainfeatureeng_refactored.py` ~1004 | `refactored_test_dir/final_processed_{m}k.csv` |
| `retrain.py` 79, 94, 95, 99 | `TRAIN`, `TEST_Q1`, `TEST_Q2`, `MODEL_DIR` |
| `milestone_features.py` 117 | the hardcoded root `traindataq1_q2/` path |

That last one is a known bug — `CLAUDE.md` records that `milestone_features.py:117` "still hardcodes
the root path" while everything else goes through `resolve_path()`. Fix it here.

## Step 4.3 — Prove it changed nothing locally

**On your laptop, before touching Fabric:**

```powershell
python retrain.py 20
```

**Pass condition:** identical metrics to your last run. `PMS_BASE` is unset and `/lakehouse` doesn't
exist, so `BASE` is `"."` and every path resolves exactly as before.

If this run differs, stop — you have a path bug, and it will be far harder to find once there are
two environments in play.

### ☑ Checkpoint 4

- [ ] `pms_paths.py` exists, uploaded to `Files/code/`
- [ ] `python retrain.py 20` on the laptop gives **identical** metrics to before
- [ ] `from pms_paths import p; print(p("data","EDA_Q2-2026.csv"))` prints the Lakehouse path in a
      Fabric notebook and a relative path on the laptop

**Safe to stop here.**

---

# Stage 5 — The feature-engineering notebook

**Goal:** `pmstrainfeatureeng_refactored.py` running in Fabric, producing a byte-identical matrix.

## Step 5.1 — Wrap `sys.argv`

The script reads `year`, `quarter`, `milestone`, `--train-cutoff`, `--label-mode`, `--label-start`,
`--date-model` from the command line. Notebooks have none of that.

Create notebook `01_feature_eng`. **First cell only**, then right-click it → **Toggle parameter cell**:

```python
# PARAMETERS  (the pipeline overrides these)
year         = 2026
quarter      = "Q1"
milestone    = 20
train_cutoff = "2025-12-31"
label_mode   = "ever"
date_model   = "schedule"
```

Second cell:

```python
import sys
sys.path.insert(0, "/lakehouse/default/Files/code")
sys.argv = ["feature_eng", str(year), quarter, str(milestone),
            "--train-cutoff", train_cutoff,
            "--label-mode", label_mode,
            "--date-model", date_model]
```

Faking `sys.argv` is a deliberate shortcut: the script's `if __name__ == "__main__":` block parses it
and you get to run the real file with **zero changes to its argument handling**. Refactor to a proper
`main(...)` later if you want; it isn't needed to migrate.

## Step 5.2 — Run the actual script

```python
%run /lakehouse/default/Files/code/pmstrainfeatureeng_refactored.py
```

If `%run` gives trouble with the `__main__` guard, use:

```python
exec(open("/lakehouse/default/Files/code/pmstrainfeatureeng_refactored.py").read(),
     {"__name__": "__main__"})
```

## Step 5.3 — `debug.txt`

`features.py` writes a progress trace to `debug.txt` in the current folder when `PMS_DEBUG` is set.
In a notebook, cwd is ephemeral — that file vanishes with the session.

**Don't edit `features.py`** (both pipelines import it). Just leave `PMS_DEBUG` unset in Fabric, or
sweep the file into `Files/run_logs/{run_id}/` at the end.

## Step 5.4 — The most important test in this guide

Run the **laptop** version and the **Fabric** version on the same milestone, and diff.

Laptop:
```powershell
python pmstrainfeatureeng_refactored.py 2026 Q1 20 --train-cutoff 2025-12-31
```

Fabric: run `01_feature_eng` with the defaults above.

Then compare `refactored_test_dir/final_processed_20k.csv` (11.6 MB, laptop) against the Fabric
output:

```python
import pandas as pd
a = pd.read_csv("/lakehouse/default/Files/derived/2026-q2/final_processed_20k.csv", low_memory=False)
b = pd.read_csv("/lakehouse/default/Files/reference/final_processed_20k.csv", low_memory=False)  # upload the laptop one here

print("rows     :", len(a), len(b), len(a) == len(b))
print("cols     :", a.shape[1], b.shape[1], list(a.columns) == list(b.columns))
num = a.select_dtypes("number").columns
print("max diff :", (a[num] - b[num]).abs().max().max())
print("positives:", a.TargetFlag.mean(), b.TargetFlag.mean())
```

**Pass condition — all four:**
- same row count
- same column names **in the same order**
- max numeric difference `< 1e-9`
- identical `TargetFlag` positive rate

**If they match:** the migration is correct. The code moved; the maths didn't.

**If they don't:** you have a porting bug, not a data change. Most likely, in order: a path resolved
to the wrong file; a `pd.read_csv` lost its `encoding='ISO-8859-1'` or `low_memory=False`; a pandas
version difference. **Do not proceed to Stage 6.**

> Do this on **milestone 20 only**. Don't run all 9 until it passes once.

### ☑ Checkpoint 5

- [ ] `01_feature_eng` completes for milestone 20
- [ ] All four parity conditions pass
- [ ] Output landed in `Files/derived/2026-q2/`, not somewhere ephemeral

**Safe to stop here.** Consider running the parity check for the other 8 milestones before moving on.

---

# Stage 6 — The training notebook

**Goal:** `retrain.py` in Fabric, with MLflow tracking the metrics you currently write to JSON.

## Step 6.1 — Same wrapper

Notebook `02_retrain`, parameter cell:

```python
milestone      = 20
seed           = 42
label_mode     = "ever"
delay_variant  = "legacy"
run_tag        = ""
```

Then:

```python
import os, sys
sys.path.insert(0, "/lakehouse/default/Files/code")
os.environ["PMS_SEED"]          = str(seed)
os.environ["PMS_LABEL_MODE"]    = label_mode
os.environ["PMS_DELAY_VARIANT"] = delay_variant
os.environ["PMS_RUN_TAG"]       = run_tag
sys.argv = ["retrain", str(milestone)]
exec(open("/lakehouse/default/Files/code/retrain.py").read(), {"__name__": "__main__"})
```

Your existing env-var switches (`PMS_SEED`, `PMS_DELAY_VARIANT`, `PMS_USE_HAS_X`, `PMS_DROP`,
`PMS_RUN_TAG`) become notebook parameters **for free**. That is the ablation harness you already
built, now driven from a pipeline instead of a shell.

## Step 6.2 — Things you must NOT change

Copied from the S3 guide's Stage 7.2 because they matter just as much here. These are hard-won
guards — if a metric looks suspiciously good after migrating, check this table first:

| Where | Guard | Why |
|---|---|---|
| `pms_model.py` `LEAK_COLS` | drops `Service_Num` + 4 others | `Service_Num == milestone` **is** the label. Exact leak, 100.00% of rows |
| `retrain.py:187-216` | the test-schema reconcile | Stops features being zero-filled at inference — your #1 metric killer |
| `retrain.py:221-224` | drops `_Cluster_` columns | Cluster "4" in training ≠ cluster "4" in test |
| `retrain.py:227` | `DROP_RFM = {60}` | RFM encodes cohort vintage, not behaviour |
| `pms_model.py` `HAS_X_EXCLUDE` | `{80}` | 80k's Q2 file has `Service_Num = 0` for all 651 positives |
| `retrain.py:237` | `IsolationForest(contamination=0.005)` | — |
| `retrain.py:248` | chronological 80/20 split | **Not random.** Order matters |
| `retrain.py:44-45` | `np.random.seed` / `torch.manual_seed` | Unseeded runs differed by 6.7 accuracy points |

## Step 6.3 — Add MLflow

Fabric tracks MLflow automatically inside a workspace. This replaces the scatter of
`metrics_*.json` files with something queryable:

```python
import mlflow
mlflow.set_experiment(f"pms-{milestone}k")
with mlflow.start_run(run_name=f"{milestone}k-{label_mode}-{delay_variant}-seed{seed}"):
    mlflow.log_params({"milestone": milestone, "seed": seed,
                       "label_mode": label_mode, "delay_variant": delay_variant})
    exec(open(".../retrain.py").read(), {"__name__": "__main__"})
    mlflow.log_metrics({"auc_q1": ..., "acc_q1": ..., "n_features": ...})
```

**This is the biggest genuine upgrade in the migration.** `CLAUDE.md` currently carries hand-typed
comparison tables — the four `PMS_Delay` variants across 20k/60k, `ever` vs `window` across 8
cohorts. Those become MLflow runs you can sort and filter, and the parameters are recorded
automatically rather than remembered.

## Step 6.4 — Watch this log line

```
Test-schema reconcile: kept N / M features
```

**On 20k, `N` should be 165.** If it reads ~129, the reconcile isn't finding your test sets and the
`family_of()` recovery didn't run. If it reads ~32, a **truncated test file** is shrinking the
model — this exact failure cost 50k about 24 accuracy points once already.

### ☑ Checkpoint 6

- [ ] `02_retrain` completes for milestone 20
- [ ] Feature count matches your laptop baseline (165 on 20k)
- [ ] Q1/Q2 metrics match the laptop to ~4 decimals
- [ ] An MLflow run appears in the workspace with params and metrics
- [ ] Artifacts under `Files/models/2026-q2/20k/`

Running it twice with the same seed must give identical metrics. If not, something is unseeded.

**Safe to stop here.**

---

# Stage 7 — The pipeline

**Goal:** all 9 milestones in one click, on a schedule.

## Step 7.1 — Build it

**New item → Data pipeline**, name `pms-refresh`.

```
[Set variable: run_id]
        |
        v
[ForEach: milestones = [20,30,40,50,60,70,80,90,100]]
        |
        +--> [Notebook: 01_feature_eng]  (params: milestone = @item())
        |
        +--> [Notebook: 02_retrain]      (params: milestone = @item())
        |
        v
[Notebook: 03_score]
```

Pass `@item()` into each notebook's `milestone` parameter through the activity's **Settings →
Base parameters**.

## Step 7.2 — Set ForEach concurrency to 2. This is the one that will bite you.

ForEach defaults to **sequential**. If you tick "parallel", it defaults to 20 concurrent — and on an
F2 that will throttle you hard.

The arithmetic: a Python notebook at the default 2 vCores burns **1 CU/second**. An **F2 has 2 CU
total**. So two notebooks saturate the entire capacity; a third queues, and interactive work in the
workspace stalls behind it.

| Capacity | Total CU | Safe parallel Python notebooks |
|---|---|---|
| F2 | 2 | 2 |
| F4 | 4 | 4 |
| F8 | 8 | 8 |
| F64 (trial) | 64 | plenty |

Fabric smooths **background** jobs (pipeline-triggered notebooks) over 24 hours, which helps — but
smoothing doesn't remove the burst, it just spreads the bill. Sequential, or 2 at a time, is right
for 9 short jobs.

## Step 7.3 — Schedule it

Your data arrives quarterly. **Do not schedule this daily** — it would burn ~90× the compute for
identical output.

Pipeline → **Schedule → Monthly**, then trigger manually when a new snapshot lands. Or leave it
manual entirely; nine milestones is one click.

### ☑ Checkpoint 7

- [ ] Pipeline runs all 9 milestones end to end
- [ ] Concurrency capped at 2
- [ ] No throttling in **Admin portal → Capacity settings → Metrics app**
- [ ] Total runtime recorded — you need it for the cost model

**Safe to stop here.**

---

# Stage 8 — Scoring, Power BI, and pausing

## Step 8.1 — Predictions as a Delta table

This is where Fabric earns its keep. Instead of `predictions/{m}k/scored_*.csv`, write a table:

```python
scored = ...   # score_milestone.py output: VIN, prob_turnup
scored["milestone"] = milestone
scored["run_id"]    = run_id
scored["scored_at"] = pd.Timestamp.utcnow()

spark.createDataFrame(scored).write.mode("append") \
     .format("delta").saveAsTable("predictions")
```

## Step 8.2 — Watch the zero-fill warning

```
WARNING: N/M features GENUINELY missing from ... and zero-filled: [...]
```

**Pass condition: `N` must be 0.**

`split_missing()` separates absent one-hot categories (harmless — 0 is the correct value) from real
pipeline gaps (which invalidate the metrics). Only the second bucket matters. This is the single
most common cause of bad numbers in this project, and migrating environments is exactly when schema
drift creeps in.

## Step 8.3 — Power BI on Direct Lake

**Lakehouse → New semantic model → select `predictions` → Create report.**

**Direct Lake** means the report reads the Delta files in OneLake directly — no import, no refresh
schedule, no copy. A service advisor's call list is current the moment the pipeline finishes.

⚠️ **The licensing trap.** On **F2–F32**, everyone who *views* that report needs a **Power BI Pro**
licence (~$14/user/month). Free viewing starts at **F64**. Ten service advisors on an F2 is
$140/month in licences on top of ~$156–263 for the capacity — more than the compute.

Budget for this before you promise anyone a dashboard. If viewers are the goal, price F64 against
Pro licences explicitly.

## Step 8.4 — Pause the capacity

**If you are paying pay-as-you-go, this is the difference between ~$1 and ~$263 a month.**

Azure portal → your Fabric capacity → **Pause**. Resume takes ~30 seconds.

Automate it so it doesn't depend on memory — Azure Automation runbook, or the pipeline itself:
resume → run → pause.

Two things that surprise people:
- **Reserved capacity cannot be paused for savings.** You pay the reservation regardless. Reserved
  is cheaper *only* if you are genuinely always-on. For a quarterly pipeline, **PAYG + pause is far
  cheaper than reserved**, despite the higher headline rate.
- **Storage keeps billing while paused.** OneLake is ~$0.023/GB/month regardless. At ~2 GB that's
  about 5 cents — ignore it.
- **A paused capacity means Power BI reports stop working.** If Stage 8.3 is in production for real
  users, you cannot pause during business hours. That changes the cost model completely, and it is
  the main reason to run the ML on AWS and keep only the report on Fabric.

### ☑ Checkpoint 8 — you're done when

- [ ] Stage 5.4 parity passed (the migration provably changed nothing)
- [ ] Scoring reports **zero** genuinely-missing features
- [ ] `predictions` Delta table populated for all 9 milestones
- [ ] MLflow holds a run per milestone with params and metrics
- [ ] Capacity pauses when idle, or you have consciously decided it stays on and priced that
- [ ] A manifest names every input file, so "which data made this model" is answerable **without
      reading any Python**

That last point is the actual goal of this whole migration — same as it was for S3.

---

# Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError` on `/lakehouse/...` | No default Lakehouse attached | Left panel → attach, check the *default* radio button |
| `ModuleNotFoundError: features` | `sys.path` missing | `sys.path.insert(0, "/lakehouse/default/Files/code")` |
| `ModuleNotFoundError: torch` | Notebook on default Environment | Toolbar → Environment → `pms-env` |
| Notebook takes 5 min to start | Custom Environment cold start | Normal. Keep the session alive while developing |
| `IndexError: list index out of range` | `sys.argv` not faked | Add the `sys.argv = [...]` cell |
| Pipeline stalls / jobs queue | ForEach concurrency > capacity CU | Cap at 2 on F2 |
| Metrics suspiciously **high** | A leak guard lost in the port | Re-check the Stage 6.2 table |
| Metrics suspiciously **low** | Features zero-filled at inference | Check the reconcile line + Stage 8.2 warning |
| Two identical runs disagree | Unseeded randomness | Check `PMS_SEED`, IsolationForest, Gower |
| Report says "capacity paused" | It's paused | Resume, or move reporting off this capacity |

---
---

# AWS_VS_FABRIC — which one is economical and sustainable

## What this workload actually is

The answer turns on this, so measure it before reading any pricing page:

| | |
|---|---|
| Raw data | **261 MB** per quarterly snapshot, 7 CSVs |
| Vehicles | 130,759 (EDA rows) |
| Feature matrices | 9–30 MB each, 9 milestones, ~150 MB total |
| Model | MLP **64→32→1**, ~165 features, ~8k training rows |
| Hardware needed | **1 CPU core.** No GPU — `torch.cuda.is_available()` is `False` today and nothing suffers |
| Full 9-milestone refresh | **~3–4 single-node CPU-hours** (feature-eng dominates; training is minutes) |
| Cadence | **Quarterly** — the data snapshots are quarterly |

**Production compute is ~1.3 hours per month.** That is the number that decides everything below.
This is not a big-data problem and you should refuse to pay for one.

## Cost, at this workload's actual size

Monthly, USD, list prices as of August 2026. Storage assumed at 2 GB steady / 10 GB with a year of
history.

| | Microsoft Fabric | AWS |
|---|---|---|
| **Storage** | OneLake $0.023/GB → **$0.05** | S3 Standard $0.023/GB → **$0.05** |
| **Compute floor** | Capacity is **always-on unless paused**. F2 PAYG $0.36/hr | **Zero.** Per-second billing, scale to zero |
| **Compute, prod only** (4 hr/quarter, paused otherwise) | **~$0.50** | SageMaker `ml.m5.xlarge` $0.23/hr → **~$0.31** |
| **Compute, realistic** (capacity up business hours, 220 hr/mo) | **~$79** | **~$0.31** (jobs don't need a live host) |
| **Compute, if you forget to pause** | F2 always-on PAYG **$263** / reserved **$156** | n/a — no such failure mode |
| **Active development** (~40 hr/mo of interactive runs) | **~$14** (F2 hourly while working) | **~$9** |
| **Orchestration** | Included in capacity | EventBridge + Step Functions **< $1** |
| **Experiment tracking** | MLflow **included** | SageMaker Experiments included, or self-host MLflow **~$15** |
| **BI for 10 viewers** | Direct Lake **+ Power BI Pro ~$14/user = $140**, or F64 (~$5,000 reserved) for free viewing | QuickSight readers **~$30**, or Power BI Pro separately |
| **Realistic all-in, no dashboard** | **$80–160** | **$10–15** |
| **Realistic all-in, 10 dashboard users** | **$220–300** | **$40–45** |
| **If your org already owns an F64** | **~$0 marginal** | still ~$10 |

**Sources:** [Fabric pricing](https://azure.microsoft.com/en-us/pricing/details/microsoft-fabric/) ·
[F-SKU guide](https://solv-systems.com/resources/microsoft-fabric-pricing-2026) ·
[OneLake consumption](https://learn.microsoft.com/en-us/fabric/onelake/onelake-consumption) ·
[SageMaker pricing](https://aws.amazon.com/sagemaker/ai/pricing/) ·
[S3 pricing](https://www.nops.io/blog/aws-s3-pricing/) ·
[notebook CU consumption](https://learn.microsoft.com/en-us/fabric/data-engineering/fabric-notebook-selection-guide)

## The structural difference in one line

**AWS's floor is zero; Fabric's floor is a capacity SKU.** For a pipeline that runs 4 hours a
quarter, that structural difference matters far more than any per-hour rate.

Fabric's counter is that the floor buys you storage + notebooks + orchestration + MLflow + a BI
layer + governance as **one product with one bill**. On AWS you assemble S3 + SageMaker/Batch +
Step Functions + EventBridge + IAM + ECR yourself.

## Side-by-side on everything else

| | Microsoft Fabric | AWS |
|---|---|---|
| **Code changes to migrate** | **Small.** OneLake mounts as a real path — `os.path.exists` and `os.makedirs` just work | **Larger.** `s3io.py` (~10 fns) + ~10 edits, because those 4 `os.path.exists` calls fail on `s3://` |
| **Pieces to learn** | 1 (workspace) | 5–6 (S3, IAM, SageMaker/Batch, Step Functions, EventBridge, CloudWatch) |
| **Scale-to-zero** | Manual pause; reserved can't pause at all | Automatic, per-second |
| **GPU if ever needed** | Higher SKU or Azure ML | Any instance, on demand |
| **BI to service advisors** | **Built in.** Direct Lake, zero integration | QuickSight, or export to Power BI anyway |
| **Fits a Microsoft shop** | Native — Entra ID, Excel, Dynamics, Teams | Extra identity + integration work |
| **Vendor lock-in** | Higher — Delta/OneLake, capacity model | Lower — S3 + plain Python containers |
| **Cost predictability** | Flat and predictable; also flat when idle | Near-zero idle; spiky under heavy dev |
| **Team fit for this repo** | Plain pandas/PyTorch runs near-unmodified in a Python notebook | Same, but you package and wire it yourself |

## "Sustainable" — both meanings

**Environmentally.** Azure and AWS both claim 100% renewable-matched electricity, so the vendor
choice is close to a wash. **The architecture is the real lever.** This workload needs ~4 CPU-hours
per quarter. An always-on F2 holds compute reserved for ~2,190 hours in that same quarter — roughly
**500× the idle footprint for identical output**. Whichever platform you pick, scale-to-zero (AWS
jobs, or Fabric with disciplined pause automation) is the greener design by a wide margin. Also:
pick one region and keep data in it — cross-region transfer is pure waste, and your data has
residency reasons to stay put anyway.

**Sustainable to operate.** Different answer. You are one engineer maintaining nine models with no
test suite, where verification means "run the eval scripts and compare metrics". Fabric's single
workspace — one bill, one identity model, MLflow and BI included, nothing to wire — is materially
less to keep alive than six AWS services. That is a genuine point in Fabric's favour and it doesn't
show up in the price table.

## The recommendation

**Ask one question first: does your organisation already own Fabric capacity (F64+) for Power BI?**

- **Yes → use Fabric.** Marginal cost is ~$0, viewers are already licensed, the code changes are
  smaller than the AWS path, and the pipeline output lands where the business already looks. Nothing
  else in this comparison outweighs a capacity someone else is already paying for.

- **No, and this stays a POC → use AWS, or stay local.** Paying $156–263/month of capacity floor for
  4 hours of quarterly compute is poor value. Honestly: at 261 MB and a 64→32→1 MLP, **your laptop
  plus S3 for durable storage** is a defensible answer, and the S3 guide already gets you there.

- **No, but it's going to production with dashboards for service advisors → Fabric, on F2 + Pro
  licences.** Once people need the output, Direct Lake removes a whole integration project and the
  comparison shifts. Price the Pro licences (Stage 8.3) before committing.

**The hybrid is worth considering and often wins:** train on AWS (or locally) where compute scales to
zero, write `predictions` into OneLake, report from Fabric. You pay Fabric only for the part that
genuinely benefits from being always-on.

## Before you commit either way

1. **Measure the real runtime.** Stage 7's checkpoint records total pipeline minutes. Every number
   above assumes ~4 hours per full refresh — confirm it.
2. **Do Stages 0–7 on the 60-day trial.** It costs nothing and answers the question with your actual
   data rather than an estimate.
3. **Decide the region once** (UAE North / `me-central-1`). VINs and nationalities are personal data,
   and the region is effectively permanent on both platforms.
4. **Count the dashboard viewers.** Above roughly 10, Power BI Pro licences dominate the bill and
   the F2-vs-F64 question becomes the real decision — not the ML compute at all.
