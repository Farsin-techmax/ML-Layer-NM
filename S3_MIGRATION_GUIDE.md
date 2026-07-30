# S3 Migration — Beginner Guide

Work through this **one stage at a time**. Every stage ends with a checkpoint and a safe stop point.
Nothing in a later stage breaks if you stop after an earlier one.

Estimated: 8 stages, 30–90 min each. Spread over as many days as you like.

---

## Progress tracker

| Stage | What you build | Done? |
|---|---|---|
| 0 | Understand the picture (no typing) | ☐ |
| 1 | Install tools, prove AWS works | ☐ |
| 2 | Create the bucket, lock it down | ☐ |
| 3 | Upload the raw data files | ☐ |
| 4 | `s3io.py` — the download/upload helper | ☐ |
| 5 | `stage_raw.py` — clean the raw files once | ☐ |
| 6 | `pmsfeatureeng_s3.py` — feature engineering | ☐ |
| 7 | `retrain_s3.py` — training | ☐ |
| 8 | Scoring + the manifest | ☐ |

---

# Stage 0 — Understand the picture

**No commands in this stage. Just read. 10 minutes.**

## The one idea that clears up most confusion

**S3 is not a hard drive.** It looks like folders when you browse it, but it isn't. It's more like a
post office box — you mail files in, you request files out. There is no "current directory", no
"create a folder", no "check if a file exists" in the way your Python code expects.

This matters because your scripts do things like:

```python
if not os.path.exists(service_history_path):   # line 955
    sys.exit(1)
```

If you hand that an S3 address like `s3://my-bucket/data/file.csv`, `os.path.exists` returns
**False** — not because the file is missing, but because it has no idea what S3 is. Your script
exits before doing anything. Same problem with `os.makedirs` (line 850, 1020) — it will crash.

There are **4 places** in your code that do this check, and they will all fail:

- `pmstrainfeatureeng_refactored.py` line 287
- `pmstrainfeatureeng_refactored.py` line 952
- `pmstrainfeatureeng_refactored.py` line 955
- `features.py` line 1071

## So what do we do?

**We don't make the scripts talk to S3 at all.**

Instead:

```
1. DOWNLOAD the files from S3 to a local folder   (a small helper does this)
2. RUN your scripts exactly as they are today      (they see normal local files)
3. UPLOAD the results back to S3                   (the same helper does this)
```

That's the whole design. Your scripts never see an `s3://` address. They keep working with normal
Windows paths, so `os.path.exists` keeps working, `os.makedirs` keeps working, nothing breaks.

The alternative — teaching every read and write in your code to speak S3 — would mean editing about
40 places and would break those 4 checks. We're doing about 10 edits instead.

## Is downloading everything slow?

No. Your whole data snapshot is **261 MB**:

| File | Size |
|---|---|
| Service_History_Q2-2026.csv | 124 MB |
| EDA_Q2-2026.csv | 52 MB |
| Appointments_Q2-2026.csv | 49 MB |
| VHC_Q2-2026.csv | 31 MB |
| Digital_Sessions_Q2-2026.csv | 2.6 MB |
| RFM_Q2-2026.csv | 1.4 MB |
| Service Code Desc.csv | 8 KB |

You download that **once**. Then all 9 milestones run off the local copy. Downloading once and
reusing is the fast path.

## What are these "layers" I keep seeing?

We split the bucket into 4 areas. Think of them as stages on an assembly line:

```
raw/       The original files, exactly as you received them.
           NEVER changed. NEVER overwritten. This is your safety net.
              |
              v
staged/    Same data, cleaned up once: dates parsed, numbers converted,
           saved as .parquet (smaller + faster to load).
              |
              v
derived/   The feature matrices your model trains on.
              |
              v
models/    The trained models: best_model.pt, scaler.joblib, etc.

run_logs/  Every run drops its log + a manifest here. Separate from the above.
```

Why bother? Because right now, when a model gives strange numbers, you have to read the source code
to find out which data files it used. After this, you read one small `manifest.json` file. That's
the real payoff.

## The one decision waiting for you

Your `data/` folder has **two versions of the same data**:

- **Legacy**: `EDA - Q3 2025.csv`, `Service History Q1 - 2026.csv`
- **New**: `EDA_Q2-2026.csv`, `Service_History_Q2-2026.csv`

Right now `pmstrainfeatureeng_refactored.py` lines 944–945 use the **legacy** EDA + service history,
but lines 946–949 use the **new** RFM/Appointments/Digital/VHC. It's a mix. That's the bug written
down in `DATA.md`.

**You do not have to decide now.** We build the pipeline so it can run either way, with a flag. You
run it on legacy first (to prove the migration didn't change anything), then flip to new and *measure*
what changed. Stage 6 covers this.

### ☑ Checkpoint 0

You can explain, out loud:
- Why we download instead of reading S3 directly → *because `os.path.exists` and `os.makedirs` don't work on S3 addresses*
- What `raw/` is for → *the untouched original, our safety net*
- What the flag in Stage 6 is for → *proving the migration changed nothing before changing the data*

**Safe to stop here.**

---

# Stage 1 — Install tools, prove AWS works

**Goal:** get to a point where one command prints your AWS identity. Nothing else.

## Step 1.1 — Install the AWS CLI

Download and run the installer: https://awscli.amazonaws.com/AWSCLIV2.msi

**Close and reopen your terminal afterwards.** (The installer changes `PATH`; existing terminals
don't see it.)

Check:

```powershell
aws --version
```

**You should see:** `aws-cli/2.x.x Python/3.x.x Windows/10 ...`

**If you see** `'aws' is not recognized` → you didn't reopen the terminal. Do that first.

## Step 1.2 — Get AWS credentials

You need an **Access Key ID** and a **Secret Access Key** from whoever owns the AWS account. Ask for
a user with S3 permissions.

⚠️ These are passwords. Never paste them into a `.py` file, never commit them to git.

## Step 1.3 — Configure a named profile

We use a named profile (`pms`) rather than the default, so this project can't accidentally touch
anything else.

```powershell
aws configure --profile pms
```

It asks 4 questions:

```
AWS Access Key ID     : (paste it)
AWS Secret Access Key : (paste it)
Default region name   : me-central-1
Default output format : json
```

**About the region:** `me-central-1` is UAE. Pick the region physically closest to you — it makes
transfers faster, and if this data has residency rules, it keeps it in-country. Once chosen,
**write it down**; changing it later means moving everything.

## Step 1.4 — Prove it works

```powershell
aws sts get-caller-identity --profile pms
```

**You should see:**

```json
{
    "UserId": "AIDA...",
    "Account": "123456789012",
    "Arn": "arn:aws:iam::123456789012:user/something"
}
```

**If you see** `InvalidClientTokenId` → the Access Key was typed wrong. Re-run `aws configure --profile pms`.
**If you see** `SignatureDoesNotMatch` → the Secret Key was typed wrong. Same fix.

## Step 1.5 — Install the Python packages

```powershell
.\venv\Scripts\activate
pip install boto3 python-dotenv pyarrow
```

(`fastparquet` is already in use by your code; `pyarrow` is a more reliable parquet writer. Having
both is fine.)

### ☑ Checkpoint 1

- [ ] `aws --version` prints a version
- [ ] `aws sts get-caller-identity --profile pms` prints your account ARN
- [ ] `pip show boto3` prints a version

**Safe to stop here.** You've touched nothing in the project yet.

---

# Stage 2 — Create the bucket and lock it down

**Goal:** an empty, private, versioned bucket.

Bucket names are **globally unique across all of AWS**. Pick something nobody else would use.
This guide assumes `techmax-cwf-pms` — substitute your own everywhere.

## Step 2.1 — Create it

```powershell
aws s3api create-bucket --bucket techmax-cwf-pms --region me-central-1 --create-bucket-configuration LocationConstraint=me-central-1 --profile pms
```

**If you see** `BucketAlreadyExists` → someone worldwide has that name. Add a suffix: `techmax-cwf-pms-2026`.
**If you see** `IllegalLocationConstraintException` → your profile's region and the `--region` flag
disagree. Make them match.

> **Note:** if you chose `us-east-1`, drop the entire `--create-bucket-configuration` argument.
> That one region is special and rejects it.

## Step 2.2 — Block all public access

**Do this immediately after creating the bucket.** Your data contains VINs, customer nationalities,
and RFM segments — personal data.

```powershell
aws s3api put-public-access-block --bucket techmax-cwf-pms --public-access-block-configuration "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true" --profile pms
```

Verify:

```powershell
aws s3api get-public-access-block --bucket techmax-cwf-pms --profile pms
```

**You should see:** all four values `true`. If any is `false`, re-run the command above.

## Step 2.3 — Turn on versioning

This means an accidental overwrite keeps the old copy. It is the single best protection against
destroying your `raw/` files.

```powershell
aws s3api put-bucket-versioning --bucket techmax-cwf-pms --versioning-configuration Status=Enabled --profile pms
```

Verify:

```powershell
aws s3api get-bucket-versioning --bucket techmax-cwf-pms --profile pms
```

**You should see:** `"Status": "Enabled"`.

## Step 2.4 — Turn on encryption at rest

```powershell
aws s3api put-bucket-encryption --bucket techmax-cwf-pms --server-side-encryption-configuration "{\"Rules\":[{\"ApplyServerSideEncryptionByDefault\":{\"SSEAlgorithm\":\"AES256\"},\"BucketKeyEnabled\":true}]}" --profile pms
```

(`AES256` is S3-managed encryption — free and automatic. KMS gives you per-key audit trails but adds
cost and setup; AES256 is the right starting point.)

### ☑ Checkpoint 2

```powershell
aws s3 ls s3://techmax-cwf-pms --profile pms
```

**You should see:** nothing at all. An empty result is success — the bucket exists and is empty.

**If you see** `NoSuchBucket` → creation failed; scroll up and re-read the error from 2.1.

**Safe to stop here.**

---

# Stage 3 — Upload the raw files

**Goal:** the 7 new-snapshot CSVs sitting in `raw/2026-q2/`, untouched forever after.

## Step 3.1 — Understand what we're skipping

Your `data/` folder is ~956 MB, but most of it is duplicates:

- `.xlsx` files (~199 MB) — byte-identical twins of the `.csv` files. **Skip.**
- Legacy-named duplicates (`Appoinments2025.csv` is the same bytes as `Appoinments - Q3 - 2025.csv`). **Skip for now.**

We upload the **7 new-snapshot CSVs only** = 261 MB.

## Step 3.2 — Dry run first

Always dry-run an `aws s3` command before the real one. It prints what *would* happen and changes
nothing.

```powershell
aws s3 sync data/ s3://techmax-cwf-pms/raw/2026-q2/ --exclude "*" --include "Service_History_Q2-2026.csv" --include "EDA_Q2-2026.csv" --include "Appointments_Q2-2026.csv" --include "VHC_Q2-2026.csv" --include "Digital_Sessions_Q2-2026.csv" --include "RFM_Q2-2026.csv" --include "Service Code Desc.csv" --profile pms --dryrun
```

**You should see:** exactly 7 lines starting with `(dryrun) upload:`.

**Count them.** If you see 8+, an `--exclude`/`--include` is wrong. If you see fewer than 7, a
filename is misspelled — compare against `data/` carefully (note `Appointments` has a **t**;
`Appoinments` without it is the legacy file).

## Step 3.3 — Do it for real

Same command, **remove `--dryrun`**:

```powershell
aws s3 sync data/ s3://techmax-cwf-pms/raw/2026-q2/ --exclude "*" --include "Service_History_Q2-2026.csv" --include "EDA_Q2-2026.csv" --include "Appointments_Q2-2026.csv" --include "VHC_Q2-2026.csv" --include "Digital_Sessions_Q2-2026.csv" --include "RFM_Q2-2026.csv" --include "Service Code Desc.csv" --profile pms
```

Takes a few minutes on a normal connection. It shows a progress line.

## Step 3.4 — Verify sizes match exactly

```powershell
aws s3 ls s3://techmax-cwf-pms/raw/2026-q2/ --human-readable --profile pms
```

**You should see 7 objects with these sizes:**

| File | Expected |
|---|---|
| Service_History_Q2-2026.csv | 118.2 MiB |
| EDA_Q2-2026.csv | 49.6 MiB |
| Appointments_Q2-2026.csv | 46.7 MiB |
| VHC_Q2-2026.csv | 29.8 MiB |
| Digital_Sessions_Q2-2026.csv | 2.5 MiB |
| RFM_Q2-2026.csv | 1.4 MiB |
| Service Code Desc.csv | 8.1 KiB |

(`aws s3 ls` reports MiB — 1 MiB = 1,048,576 bytes — so these look smaller than the MB figures in
Stage 0. Same bytes, different unit.)

**If a size is wrong** → re-run the sync; it re-uploads only mismatched files.

### ☑ Checkpoint 3

- [ ] 7 objects under `raw/2026-q2/`
- [ ] All sizes match the table
- [ ] You have written down your bucket name and region somewhere

**Safe to stop here — and this is a genuinely useful stopping point.** Even with nothing else done,
your raw data now has an off-machine, versioned backup.

---

# Stage 4 — `s3io.py`, the helper

**Goal:** one small file that every other script imports. This is the only file that knows S3 exists.

Do **not** touch any existing project file in this stage. You are only adding.

## Step 4.1 — Create the `.env` file

At the repo root, create a file named exactly `.env`:

```
PMS_BUCKET=techmax-cwf-pms
PMS_SNAPSHOT=2026-q2
PMS_REGION=me-central-1
PMS_CACHE=.s3cache
PMS_PUSH=1
AWS_PROFILE=pms
```

**Then add both to `.gitignore` immediately:**

```
.env
.s3cache/
```

`.env` holds your configuration. It holds **no passwords** — the AWS keys stay in the profile you set
up in Stage 1. Even so, don't commit it.

What each one does:

| Variable | Meaning |
|---|---|
| `PMS_BUCKET` | Your bucket name |
| `PMS_SNAPSHOT` | Which data vintage. Change this when new data arrives; everything else follows |
| `PMS_CACHE` | Local folder where downloads land. Safe to delete anytime — it re-downloads |
| `PMS_PUSH` | `1` = upload results. `0` = compute but upload nothing (practice mode) |
| `AWS_PROFILE` | The profile from Stage 1 |

## Step 4.2 — What `s3io.py` must contain

Build it in this order. **Test each function before writing the next one.**

**1. `_load_env()`** — read `.env` via `python-dotenv`. Raise a clear error naming the missing
variable if `PMS_BUCKET` is absent. Runs automatically on import.

**2. `selftest()`** — prints bucket, snapshot, region, cache path, and the caller ARN from
`sts.get_caller_identity()`. This is your "is anything plugged in" command.

**3. `resolve(layer, name) -> Path`** — *the most important function.*

- Builds the local path: `{PMS_CACHE}/{layer}/{PMS_SNAPSHOT}/{name}`
- If the file isn't there → download it from `s3://{bucket}/{layer}/{snapshot}/{name}`
- If it *is* there → do a `head_object` call, compare ETag and size against a saved
  `.meta.json` sidecar. Re-download only on mismatch.
- **Returns a normal local Windows path.** Never an `s3://` string.

That last line is the point of the entire design. Everything downstream receives a real path and
keeps working.

**4. `push(local_path, layer, name)`** — upload one file. When `PMS_PUSH=0`, log the intended
destination and return without uploading.

**5. `push_dir(local_dir, layer, prefix)`** — upload a whole folder. Needed for model artifacts
(11 files per milestone).

**6. `RAW_FILES`** — a dictionary mapping short names to real filenames. This is where the
legacy-vs-new decision lives, in **one place**:

```
"service_history" -> "Service_History_Q2-2026.csv"
"eda"             -> "EDA_Q2-2026.csv"
"appointments"    -> "Appointments_Q2-2026.csv"
"vhc"             -> "VHC_Q2-2026.csv"
"digital"         -> "Digital_Sessions_Q2-2026.csv"
"rfm"             -> "RFM_Q2-2026.csv"
"servcode"        -> "Service Code Desc.csv"
```

Add a second dictionary `RAW_FILES_LEGACY` with the legacy EDA and service-history names, selected by
a `PMS_VINTAGE` variable. Stage 6 needs this.

**7. `run_id(tag)`** — returns `PMS_RUN_ID` if set, else `20260730T143000Z-20k`.

**8. `attach_run_log(run_id)`** — adds a `logging.FileHandler`, **and** tees `sys.stdout` to the same
file (your feature-eng script uses plain `print()` heavily — lines 941, 977, 987, 1014, 1026, 1031,
1036 — and those would otherwise be lost). Registers an `atexit` hook to upload the log.

> Register the atexit hook **before** any real work starts. A crashed run needs its log uploaded
> more than a successful one does.

**9. `write_manifest(run_id, info)`** — writes and uploads `run_logs/{run_id}/manifest.json`.

**10. `load(path, columns=None)`** — reads `.csv` or `.parquet` based on the file extension.
Stage 6 uses this so the same code works before and after you switch to parquet.

## Step 4.3 — Test it

```powershell
python -c "import s3io; s3io.selftest()"
```

**You should see:** bucket, snapshot, cache dir, and your ARN. No traceback.

Then test `resolve`:

```powershell
python -c "import s3io; print(s3io.resolve('raw', 'RFM_Q2-2026.csv'))"
```

**You should see:** a local path like `.s3cache\raw\2026-q2\RFM_Q2-2026.csv`, and that file now
exists on disk (1.4 MB). Start with RFM — it's the smallest, so a mistake costs 2 seconds not 3
minutes.

Run the exact same command a second time.

**You should see:** the same path, printed instantly, with no download. That proves the cache works.

### ☑ Checkpoint 4

- [ ] `selftest()` prints your ARN
- [ ] `resolve('raw', 'RFM_Q2-2026.csv')` downloads the file and returns a local path
- [ ] Running it again is instant (cached)
- [ ] `.env` and `.s3cache/` are in `.gitignore`

**Safe to stop here.** No existing project file has been modified.

---

# Stage 5 — `stage_raw.py`

**Goal:** read each raw CSV once, clean it, save as parquet, upload to `staged/`.

## Step 5.1 — Why this stage exists

Two reasons, both concrete:

**Reason 1 — the same file is read twice per run.** `pmstrainfeatureeng_refactored.py` reads
`Service_History` at line 119 *and again* at line 322. It's a 124 MB CSV, and parsing dates on it is
slow. Across 9 milestones that's 18 full parses of the same file. Staging does the work once.

**Reason 2 — there's an inconsistency worth fixing.** Line 119 reads it with
`encoding='ISO-8859-1'`. Line 322 reads the same file with **no encoding argument at all**. Those can
disagree on rows containing non-English characters. After staging, both read the identical parquet
file, so the inconsistency disappears.

⚠️ That second one is a **real behaviour change**. Stage 6's test is designed to measure it rather
than let it surprise you.

## Step 5.2 — What the script does per file

**Service history** (the important one):

1. `resolve('raw', 'Service_History_Q2-2026.csv')`
2. Read with `encoding="ISO-8859-1", low_memory=False`
3. `Service_Date` → `pd.to_datetime(..., format='mixed', errors='coerce')`
4. `Mileage` → `pd.to_numeric(..., errors='coerce')`
5. `Service_Num` → `Description.apply(extract_kk)` (import from `pmstrainfeatureeng_refactored`)
6. Write `.s3cache/staged/2026-q2/service_history.parquet`
7. `push(...)`

**The other six**: read the way their consumer reads them today, write parquet, push.

One detail — `servcode` is read with `skipinitialspace=True` at line 312. Keep that.

## Step 5.3 — The safety rule

**Before writing any parquet file, compare row counts.** If the staged file has a different number
of rows than the raw CSV, **stop and write nothing.**

A silently dropped row here would corrupt every model downstream and would be nearly impossible to
trace back. This check costs one line.

## Step 5.4 — The manifest

Write `staged/2026-q2/_manifest.json` recording per file: source filename, S3 `version_id`, `ETag`,
row count, column count, and dtypes.

This is the file that answers "which data did this model actually see" without reading any Python.

## Step 5.5 — Run it

```powershell
python stage_raw.py --verify-only
```

**You should see:** a table of every file with raw rows vs staged rows.
**Pass condition:** every difference is **0**.

Then for real:

```powershell
python stage_raw.py
```

## Step 5.6 — Verify types survived

This is the most important check in the whole guide. Everything downstream depends on `Service_Num`.

```powershell
python -c "import pandas as pd, s3io; d=pd.read_parquet(s3io.resolve('staged','service_history.parquet')); print('rows', len(d)); print('date nulls', d.Service_Date.isna().sum()); print('mileage nulls', d.Mileage.isna().sum()); print(d.Service_Num.value_counts().head(12))"
```

Now compute the same three numbers from the raw CSV and compare:

```powershell
python -c "import pandas as pd, s3io; from pmstrainfeatureeng_refactored import extract_kk; d=pd.read_csv(s3io.resolve('raw','Service_History_Q2-2026.csv'), low_memory=False, encoding='ISO-8859-1'); d['Service_Date']=pd.to_datetime(d.Service_Date, format='mixed', errors='coerce'); d['Mileage']=pd.to_numeric(d.Mileage, errors='coerce'); d['Service_Num']=d.Description.apply(extract_kk); print('rows', len(d)); print('date nulls', d.Service_Date.isna().sum()); print('mileage nulls', d.Mileage.isna().sum()); print(d.Service_Num.value_counts().head(12))"
```

**Pass condition:** the two outputs are **identical**. Row count, both null counts, and the
`Service_Num` distribution.

**If `Service_Num` differs** → stop. Do not continue to Stage 6. Something in the staging transform
is wrong, and every model built on it would be wrong too.

## Step 5.7 — Check the size win

```powershell
aws s3 ls s3://techmax-cwf-pms/staged/2026-q2/ --human-readable --profile pms
```

**You should see:** the same 7 files, much smaller — expect roughly 40–70 MiB total versus 248 MiB
of CSV. Exact numbers depend on your data; just note them.

### ☑ Checkpoint 5

- [ ] `--verify-only` shows zero row differences on all 7 files
- [ ] Service history type check matches raw exactly
- [ ] `staged/2026-q2/` has 7 parquet files + `_manifest.json`
- [ ] Re-running `stage_raw.py` skips everything (ETag match)

**Safe to stop here.**

---

# Stage 6 — `pmsfeatureeng_s3.py`

**Goal:** feature engineering reading from `staged/`, writing to `derived/`.

⚠️ **Copy the file. Do not edit `pmstrainfeatureeng_refactored.py`.** Keeping the original working
means you can always compare against it — which is exactly what Step 6.5 does.

```powershell
Copy-Item pmstrainfeatureeng_refactored.py pmsfeatureeng_s3.py
```

## Step 6.1 — Swap the input paths

Find lines 944–950 (inside `if __name__ == "__main__":`) and replace the hardcoded paths with
`s3io.resolve('staged', ...)` calls.

**Leave the `os.path.exists` checks at lines 952 and 955 exactly as they are.** They now receive real
local paths and work correctly. This is the payoff from Stage 0's design decision.

## Step 6.2 — Make the reads format-aware

Eight places read a file directly. Change each to `s3io.load(path)`:

| Line | Reads |
|---|---|
| 104 | EDA |
| 119 | service history (1st read) |
| 312 | servcode |
| 322 | service history (2nd read) |
| 578 | VHC |
| 592 | appointments |
| 622 | RFM |
| 813 | digital sessions |

`load()` picks CSV or parquet by extension, so this works whether you point it at `raw/` or
`staged/`. Drop the now-redundant `encoding=` arguments when reading parquet — parquet stores its own
types.

## Step 6.3 — Redirect the outputs

Writes go to a local run folder first, then get pushed. Map them like this:

| Currently writes | New destination |
|---|---|
| line 851 `models/{m}k/training_features.csv` | `derived/{snap}/train/{m}k/{year}{Q}.csv` |
| line 852 `validatecode/finalmerged{m}kQ3.csv` | **delete this line** — see below |
| line 1004 `refactored_test_dir/final_processed_{m}k.csv` | `derived/{snap}/full/{m}k/{year}{Q}.csv` |
| line 1025 `{m}k/training/training_features.csv` | `derived/{snap}/train_selected/{m}k/{year}{Q}.csv` |
| line 1030 `mi_scores.csv` | `run_logs/{run_id}/mi_scores.csv` |
| line 1035 `selected_features.json` | `derived/{snap}/train_selected/{m}k/selected_features.json` |
| line 355 `validatecode/serv_filteredpmsapp.csv` | `run_logs/{run_id}/debug/` |
| lines 810–812 the 3 cluster CSVs | `run_logs/{run_id}/debug/` |

**About deleting line 852:** that line writes `finalmerged{m}kQ3.csv`, which you then have to manually
copy and rename into `traindataq1_q2/` before training — and the required name is inconsistent
(`Q2026v1` for 20k/30k, but `Q2_2026v1` for 40k and up, see `retrain.py` lines 59–62). The new
`{year}{Q}.csv` naming removes both the manual copy and the inconsistency. Stage 7 deletes that
`if MILESTONE in [20, 30]` branch entirely.

**A hazard this also fixes:** because `is_test` only controls a row filter (line 307), running the
test-set builder today **overwrites** your training files at lines 851–852. Splitting `train/` from
`test/` in the path makes that impossible.

## Step 6.4 — Add logging

At the top of `main`: call `s3io.attach_run_log(run_id)` **before** anything else. At the end, call
`write_manifest` with milestone, year, quarter, cutoff, input ETags, output keys, row counts, and the
positive rate.

Also: `features.py` appends to `debug.txt` in the current folder (6 places, lines 178–219).
**Don't edit `features.py`** — both pipelines import it. Just sweep `debug.txt` into
`run_logs/{run_id}/debug/` at the end.

## Step 6.5 — The most important test in this guide

Run the **original** script and the **new** script on the **same legacy data**, and diff the outputs.

```powershell
python pmstrainfeatureeng_refactored.py 2026 Q2 20 --train-cutoff 2025-12-31
```

```powershell
python pmsfeatureeng_s3.py 2026 Q2 20 --train-cutoff 2025-12-31 --vintage legacy
```

Then compare `refactored_test_dir/final_processed_20k.csv` against the new
`derived/2026-q2/full/20k/2026Q2.csv`.

**Pass condition — all four:**
- Same number of rows
- Same set of column names
- Same column order
- Every numeric difference under `1e-9`

**If they match:** your migration is correct. The code moved; the maths didn't. You can trust
everything after this.

**If they don't match:** you have a porting bug, not a data change. Do not proceed. The most likely
causes, in order: a `load()` call dropped an argument the original had (`encoding`,
`skipinitialspace`, `low_memory`); a path resolved to the wrong vintage; a write went to the wrong
place. Fix it before Stage 7.

> Do this on **milestone 20 only**. Don't run all 9 until this passes once.

## Step 6.6 — Now measure the vintage change

Only after 6.5 passes:

```powershell
python pmsfeatureeng_s3.py 2026 Q2 20 --train-cutoff 2025-12-31 --vintage q2-2026
```

Write down, side by side with the 6.5 run:

| | legacy | q2-2026 |
|---|---|---|
| rows | | |
| columns | | |
| TargetFlag positive rate | | |

**These will differ.** That's expected and correct — you just fixed the mixed-vintage bug from
`DATA.md`. The point of doing it in this order is that you now know *exactly* how much of any change
came from the migration (zero, proven in 6.5) versus from the data (this table).

### ☑ Checkpoint 6

- [ ] Step 6.5 matches exactly on milestone 20
- [ ] `derived/2026-q2/` contains no filename with `finalmerged`, `Q2026v1`, or `Q3` in it
- [ ] `models/20k/` and `validatecode/` were **not** modified by the new script (check with `git status` and file timestamps)
- [ ] The vintage comparison table above is filled in
- [ ] A log file exists under `run_logs/` in S3

**Safe to stop here.** Consider running 6.5 for the other 8 milestones before moving on.

---

# Stage 7 — `retrain_s3.py`

**Goal:** train from `derived/`, save to `models/`.

```powershell
Copy-Item retrain.py retrain_s3.py
```

## Step 7.1 — Replace the config block

Lines 58–66 currently hardcode filenames. Replace with globs:

- Training: every `.csv` under `derived/{snap}/train/{m}k/`, concatenated
- Test: every `.csv` under `derived/{snap}/test/{m}k/`
- Model output: local `{PMS_CACHE}/models/{snap}/{m}k/`, pushed to `models/{snap}/{m}k/` after

The `if MILESTONE in [20, 30]` branch at lines 59–62 gets **deleted**. It only existed because of
inconsistent filenames, and Stage 6 fixed those.

## Step 7.2 — Things you must NOT change

These are hard-won guards. Copy them across untouched:

| Line | Guard | Why it's there |
|---|---|---|
| 106 | drops `Service_Num` + 4 others | `Service_Num == milestone` **is** the label. Exact leak |
| 120–134 | `_test_feature_cols` intersection | Stops features being zero-filled at inference — your #1 metric killer |
| 143 | drops `_Cluster_` columns | Cluster "4" in training ≠ cluster "4" in test |
| 149 | `DROP_RFM = {60}` | RFM encodes cohort age, not behaviour |
| 48 | `HAS_X_EXCLUDE = {80}` | 80k's Q2 file has `Service_Num = 0` for all 651 positives |
| 159 | `IsolationForest(contamination=0.005)` | — |
| 167 | chronological 80/20 split | Not random — order matters here |

If a metric looks suspiciously good after training, check this table first.

## Step 7.3 — One improvement worth making

Line 89 hardcodes `2025-12-31` as the training cutoff. Make it a `--train-cutoff` argument defaulting
from the manifest. Otherwise, when you load a 2026-Q3 snapshot later, it will silently ignore all the
new data and you'll wonder why nothing changed.

## Step 7.4 — Run it

```powershell
python retrain_s3.py 20
```

Watch the log for this line:

```
Test-schema intersection: kept N / M features (dropped M-N train-only cols)
```

**If `dropped` is 0**, the intersection didn't find your test files — the glob path is wrong. Fix
before trusting any metric.

## Step 7.5 — Compare against your current baseline

Look at `models/models_alan/20k/metrics_20k_Q1_Test.json` and compare with the new run.

⚠️ **Set expectations correctly:** these will **not** match, and that's not a bug. Your existing
`tests_alan/traindataq1_q2/finalmerged20kQ2026v1.csv` has 254 columns, but the current code produces
203. That training file was made by an older version of the code and **cannot be reproduced** by
running it today.

So this run establishes a **new baseline**. Write the numbers down. From here on, reproducibility is
what you check.

### ☑ Checkpoint 7

- [ ] `retrain_s3.py 20` completes
- [ ] The intersection line reports a non-zero drop count
- [ ] 11 artifacts appear under `models/2026-q2/20k/` in S3
- [ ] New baseline metrics written down
- [ ] Running it twice gives identical `selected_features.json` and identical metrics to 4 decimals

That last one matters — if two identical runs disagree, something is unseeded (check the Gower
clustering and IsolationForest seeds).

**Safe to stop here.**

---

# Stage 8 — Scoring and wrap-up

## Step 8.1 — Score a cohort

```powershell
python score_milestone.py 20 --test .s3cache\derived\2026-q2\test\20k\2026Q3.csv --model-dir .s3cache\models\2026-q2\20k
```

**Watch for this warning:**

```
WARNING: N/M features absent from the test file and zero-filled: [...]
```

**Pass condition: N must be 0.** If it lists features, your test file's columns don't match what the
model was trained on, and the probabilities are meaningless. That is the single most common cause of
bad numbers in this project.

## Step 8.2 — A note on future quarters

If you score a window in the future (say Q3 2026), the `TargetFlag` column will be **almost all
zeros** — not because those vehicles won't turn up, but because `prepare_test_set.py` (lines 86–92)
labels from service history that ends around Q2 2026. The outcome hasn't happened yet.

So for a genuine future quarter: **score only, ignore metrics.** `score_milestone.py` already detects
this and prints `TargetFlag is constant -- metrics undefined`. That message is correct behaviour, not
an error.

## Step 8.3 — Test the crash path

Start a run and press **Ctrl-C** partway through.

**Pass condition:** the log still appears in S3 under `run_logs/{run_id}/run.log`, containing the
partial output and the interrupt.

A pipeline that only uploads logs on success gives you nothing on the days you actually need them.

## Step 8.4 — Practice mode

Set `PMS_PUSH=0` in `.env` and run a full sweep.

**Pass condition:** everything completes, the log lists every upload it *would* have made, and
`aws s3 ls` shows no new objects. Use this mode whenever you're unsure.

### ☑ Checkpoint 8 — you're done when

- [ ] Stage 6.5 parity passed (the migration provably changed nothing)
- [ ] Staging conserves rows and types exactly
- [ ] Scoring reports **zero** zero-filled features
- [ ] A crashed run still uploads its log
- [ ] `manifest.json` names every input file with its version, so "which data made this model" is answerable from S3 alone — **without reading any Python**

That last point is the actual goal of this whole migration.

---

# Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `'aws' is not recognized` | Terminal opened before install | Close and reopen the terminal |
| `InvalidClientTokenId` | Access Key wrong | `aws configure --profile pms` |
| `SignatureDoesNotMatch` | Secret Key wrong | `aws configure --profile pms` |
| `NoSuchBucket` | Typo, or wrong region | Check name and `--profile pms` |
| `AccessDenied` | Your IAM user lacks S3 rights | Ask the account owner for `s3:GetObject`, `s3:PutObject`, `s3:ListBucket` |
| `Error: Could not find data/...` | Path resolution failed | Print the resolved path — it should be a local `.s3cache\...` path, never `s3://` |
| Script exits instantly, no message | An `os.path.exists` got an `s3://` string | `resolve()` must return a local path |
| `FileNotFoundError` on `os.makedirs` | Tried to create a folder in S3 | Directories are local only; S3 gets `push()` |
| Metrics suspiciously high | A leak guard was dropped in the copy | Re-check the Step 7.2 table |
| Metrics suspiciously low | Features zero-filled at inference | Check the intersection log line and the scoring warning |
| Two identical runs disagree | Unseeded randomness | Check Gower clustering + IsolationForest seeds |

---

# Glossary

**Bucket** — the top-level container in S3. Globally unique name.

**Key** — the full path of an object inside a bucket, e.g. `raw/2026-q2/EDA_Q2-2026.csv`. It looks
like folders but it's really one long name.

**Prefix** — the leading part of a key, e.g. `raw/2026-q2/`. What S3 uses instead of folders.

**ETag** — a fingerprint of an object's contents. Two files with the same ETag are the same file.
Used here to skip pointless re-downloads.

**Versioning** — keeps old copies when a file is overwritten. Your undo button.

**Parquet** — a column-based file format. Smaller than CSV, faster to load, and remembers column
types so dates don't need re-parsing.

**Manifest** — a small JSON file recording exactly what a run used and produced.

**Snapshot** — one complete vintage of the data (`2026-q2`). Changing `PMS_SNAPSHOT` in `.env`
redirects the whole pipeline to a different vintage.
