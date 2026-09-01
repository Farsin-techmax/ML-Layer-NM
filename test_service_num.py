"""test_service_num.py -- is the PMS service number read correctly, everywhere?

Plain script (no pytest in this repo): `python test_service_num.py`. Exit 0 = all pass.

BACKGROUND
`Description` is a mileage band whose upper bound is the service number, but the same 20k service
is spelled three different ways across the files:

    '11-20'   clean            -- the July 2026 extract
    'Nov-20'  Excel damage     -- main history, everywhere EXCEPT Q1 2026
    '11_20'   a manual repair  -- main history, Q1 2026 ONLY (1,967 rows)

`extract_kk` reads the first two as 20 and the third as **1120**, because Python's int() treats '_'
as a digit separator (PEP 515). Q1 2026 is exactly the window every 20k test set is built on, so
481 of 951 vehicles in that cohort were labelled "did not turn up" when they had.

`Service_Code` is a plain integer and cannot be misread, so it is now the source of truth and
Description is only the fallback -- `features.resolve_service_num()`.

THE CHECKS
  A. UNIT      -- resolve_service_num on hand-built rows covering all three spellings, the
                  non-numeric branch codes, '<=10', and 'Others'.
  B. INVARIANT -- the 0 = non-PMS convention must survive. 12 places in features.py test
                  `Service_Num == 0`; if 'Others' started resolving to its real code (5/15/25...)
                  every non-PMS feature would silently empty out. This is the check that would
                  catch a naive "just use Service_Code" change.
  C. FILE      -- run it over the whole real history and assert the diff against the old rule is
                  EXACTLY the intended set and nothing else.
  D. CALLERS   -- every site that derives Service_Num uses the shared helper.
  E. COHORT    -- the end-to-end consequence: the 20k Q1 cohort's positive rate.
"""
import importlib
import sys

import numpy as np
import pandas as pd

from features import extract_kk, resolve_service_num, PMS_SERVICE_NUMS

RESULTS = []
HIST = "data/Service_History_Q2-2026.csv"


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    return ok


def section_a_unit():
    print("\n=== A. UNIT: resolve_service_num on known rows ===")
    df = pd.DataFrame({
        "Description": ["11-20", "Nov-20", "11_20", "<=10", "21-30", "Others", "1",
                        "Others", "91-100"],
        "Service_Code": ["20", "20", "20", "10", "30", "15", "1", "GM", "100"],
    })
    got = resolve_service_num(df).tolist()
    want = [20, 20, 20, 10, 30, 0, 1, 0, 100]
    check("all three 20k spellings resolve to 20", got[:3] == [20, 20, 20], f"{got[:3]}")
    check("'<=10' resolves to 10 (no hand patch needed)", got[3] == 10, f"{got[3]}")
    check("'Others' with a non-milestone code -> 0", got[5] == 0, f"{got[5]}")
    check("'Others' with a non-numeric code -> 0", got[7] == 0, f"{got[7]}")
    check("first service stays 1", got[6] == 1, f"{got[6]}")
    check("full row-by-row expectation", got == want, f"{got}")

    # The exact trap, stated on its own so a regression names itself.
    check("int('11_20') is 1120 -- the bug this guards", int("11_20") == 1120)
    check("bare extract_kk still misreads '11_20' (helper is what fixes it)",
          extract_kk("11_20") == 1120 and resolve_service_num(
              pd.DataFrame({"Description": ["11_20"], "Service_Code": ["20"]}))[0] == 20)

    # Fallback: Service_Code absent entirely.
    only_desc = pd.DataFrame({"Description": ["11_20", "Nov-20", "<=10", "Others"]})
    got = resolve_service_num(only_desc).tolist()
    check("Description-only fallback still fixes '11_20'", got == [20, 20, 10, 0], f"{got}")


def section_b_invariant():
    print("\n=== B. INVARIANT: 0 still means non-PMS ===")
    df = pd.DataFrame({
        "Description": ["Others"] * 6,
        "Service_Code": ["5", "15", "25", "35", "45", "55"],
    })
    got = resolve_service_num(df).tolist()
    check("intermediate codes (5,15,25,...) resolve to 0, NOT to themselves",
          got == [0] * 6,
          f"{got} -- if these became 5/15/25 every `Service_Num == 0` feature would empty out")
    check("PMS_SERVICE_NUMS excludes the intermediates",
          not (PMS_SERVICE_NUMS & {5, 15, 25, 35, 45, 55}),
          f"{sorted(PMS_SERVICE_NUMS & {5, 15, 25, 35, 45, 55})}")
    check("PMS_SERVICE_NUMS covers 1 and every 10k step to 400",
          {1, 10, 20, 100, 400}.issubset(PMS_SERVICE_NUMS))


def section_c_file():
    print("\n=== C. FILE: whole history, diff vs the old rule ===")
    s = pd.read_csv(HIST, low_memory=False, encoding="ISO-8859-1",
                    usecols=["Vin_No", "Description", "Service_Code"])
    old = s["Description"].apply(extract_kk)
    old = pd.Series(np.where(s["Description"] == "<=10", 10, old), index=s.index)
    new = resolve_service_num(s)

    diff = old != new
    changed = s[diff].assign(old=old[diff], new=new[diff])

    # Intended change 1: the Q1-2026 '11_20' rows.
    u = changed[changed["Description"] == "11_20"]
    check("every '11_20' row moves 1120 -> 20",
          len(u) == 1967 and set(u["old"]) == {1120} and set(u["new"]) == {20},
          f"{len(u)} rows")

    # Intended change 2: rows whose Description is uninformative but whose code names a milestone.
    o = changed[changed["Description"] == "Others"]
    check("'Others' rows only move 0 -> a real milestone",
          set(o["old"]) <= {0} and set(o["new"]).issubset(PMS_SERVICE_NUMS),
          f"{len(o)} rows -> {sorted(set(o['new']))}")

    check("NOTHING else changed",
          set(changed["Description"]) <= {"11_20", "Others"},
          f"unexpected: {sorted(set(changed['Description']) - {'11_20', 'Others'})}")
    check("change is surgical (<0.5% of rows)", diff.mean() < 0.005,
          f"{int(diff.sum()):,}/{len(s):,} = {100*diff.mean():.3f}%")

    # The non-PMS population must be essentially untouched.
    drop = int((old == 0).sum()) - int((new == 0).sum())
    check("non-PMS row count barely moves", 0 <= drop <= 50,
          f"{int((old==0).sum()):,} -> {int((new==0).sum()):,} ({drop:+,})")
    check("20k rows increase by the recovered 11_20 rows",
          int((new == 20).sum()) - int((old == 20).sum()) == 1969,
          f"{int((old==20).sum()):,} -> {int((new==20).sum()):,}")
    return s, new


def section_d_callers():
    print("\n=== D. CALLERS: every derivation site uses the helper ===")
    files = ["prepare_test_set.py", "pmstrainfeatureeng_refactored.py", "milestone_features.py",
             "july_candidates_overlap.py", "july_overlap_diagnose.py"]
    for f in files:
        src = open(f, encoding="utf-8").read()
        bad = [ln.strip() for ln in src.splitlines()
               if "Service_Num" in ln and "=" in ln and not ln.strip().startswith("#")
               and ("extract_kk" in ln or "extract_k(" in ln
                    or 'to_numeric(serv["Service_Code"]' in ln
                    or "to_numeric(serv['Service_Code']" in ln)]
        check(f"{f} derives Service_Num only via resolve_service_num", not bad,
              "; ".join(bad)[:110])
        # Import the module for real -- a missing import is invisible to a text scan and only
        # surfaces as a NameError at runtime, which is exactly how milestone_features.py shipped
        # broken after the date fix.
        try:
            mod = importlib.import_module(f[:-3])
            names = [n for n in ("parse_dates", "resolve_service_num") if n in src]
            missing = [n for n in names if not hasattr(mod, n)]
            check(f"{f} imports resolve exactly", not missing, f"missing: {missing}")
        except Exception as exc:
            check(f"{f} imports resolve exactly", False, f"{type(exc).__name__}: {exc}"[:110])

    # The hand-rolled '<=10' patch is now inside the helper; a leftover copy would double-apply.
    for f in ["pmstrainfeatureeng_refactored.py", "milestone_features.py"]:
        src = open(f, encoding="utf-8").read()
        check(f"{f} no longer hand-patches '<=10'", "'<=10', 10," not in src)


def section_e_cohort():
    print("\n=== E. COHORT: the end-to-end consequence on 20k Q1 2026 ===")
    import prepare_test_set as pts
    cohort = pts.create_test_cohort(
        eda_path="data/EDA_Q2-2026.csv", service_history_path=HIST, milestone=20,
        start_date="2026-01-01", end_date="2026-03-31", cutoff_date="2025-12-31",
        use_mileage_projection=False, date_model="schedule")
    rate = 100 * cohort["TargetFlag"].mean()
    check("20k Q1 positive rate is back near the documented ~79%", 70 <= rate <= 85,
          f"{int(cohort['TargetFlag'].sum())}/{len(cohort)} = {rate:.1f}%")
    return cohort


def main():
    print("=" * 78)
    print("Service_Num resolution test suite")
    print("=" * 78)
    section_a_unit()
    section_b_invariant()
    section_c_file()
    section_d_callers()
    section_e_cohort()

    n = len(RESULTS)
    bad = [r for r, ok in RESULTS if not ok]
    print("\n" + "=" * 78)
    print(f"{n - len(bad)}/{n} passed")
    for r in bad:
        print(f"  FAILED: {r}")
    print("=" * 78)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
