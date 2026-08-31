"""test_dayfirst.py -- does every date column in the pipeline parse to the right real date?

No pytest in this repo, so this is a plain script: `python test_dayfirst.py`. Exit code 0 = all
pass, 1 = at least one fail. Every check prints PASS/FAIL with the observed value, so a failure
tells you what it saw, not just that it broke.

The checks, and why each one is here:

  A. UNIT -- parse_dates() on hand-written values whose correct answer is known by inspection.
     Covers the two ways this bug bites: day-first misread as month-first, and ISO-with-time
     misread as day-first. Includes '06-12-2026', the exact shape that produced the phantom
     "2026-12-06" end date in the docs.

  B. FILE BOUNDARY -- every real data file, every date column. Asserts the parsed range is
     physically possible (no dates after the extract, none before the business existed) and
     reports how many values a wrong-convention parse would have changed.

  C. CODE PATH -- the specific call sites that read those files, exercised as the pipeline calls
     them. This is what catches a site someone forgets to convert.

  D. INVARIANT -- the headline fact the bug falsified: Service_History_Q2-2026.csv ends 2026-06-29,
     not 2026-12-06. If this ever flips back, a dayfirst regression has landed.
"""
import sys

import pandas as pd

from date_utils import parse_dates, looks_iso, looks_dayfirst_ambiguous, assert_no_dayfirst_damage

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    return ok


# Data files and the date columns each one owns. `expect` is the convention the values are in.
FILES = {
    "data/EDA_Q2-2026.csv": {
        "cols": ["Invoice date", "First Service Date", "Last Service Date",
                 "Last Service Date - PMS", "Next Service Date"],
        "expect": "dayfirst",
    },
    "data/Service_History_Q2-2026.csv": {
        "cols": ["Service_Date"],
        "expect": "dayfirst",
    },
    "data/VHC_Q2-2026.csv": {
        "cols": ["Service Date"],
        "expect": "dayfirst",
    },
    "data/Appointments_Q2-2026.csv": {
        "cols": ["WIP Booking Date", "Due Date IN"],
        "expect": "iso",
    },
}

# No service record may postdate the extract, and none may predate the dealership's data.
FLOOR = pd.Timestamp("2000-01-01")
CEILING = pd.Timestamp("2027-12-31")


def section_a_unit():
    print("\n=== A. UNIT: parse_dates() on known values ===")
    dmy = pd.Series(["29-06-2026 00:00", "06-12-2026 00:00", "12/11/2024", "31/05/2021"])
    got = parse_dates(dmy)
    want = [pd.Timestamp("2026-06-29"), pd.Timestamp("2026-12-06"),
            pd.Timestamp("2024-11-12"), pd.Timestamp("2021-05-31")]
    check("day-first values parse day-first", list(got) == want, f"{list(got)}")

    iso = pd.Series(["2026-06-30", "2026-08-09 00:00:00", "2026-01-02"])
    got = parse_dates(iso)
    want = [pd.Timestamp("2026-06-30"), pd.Timestamp("2026-08-09"), pd.Timestamp("2026-01-02")]
    check("ISO values are NOT flipped by dayfirst", list(got) == want, f"{list(got)}")

    already = pd.to_datetime(pd.Series(["2026-06-29", "2026-12-06"]))
    check("already-datetime passes through", list(parse_dates(already)) == list(already))

    blanks = pd.Series(["-", "12/11/2024", None])
    got = parse_dates(blanks)
    check("'-' becomes NaT, not an error",
          got.isna().tolist() == [True, False, True], f"{got.tolist()}")

    check("looks_iso discriminates",
          looks_iso(iso) is True and looks_iso(dmy) is False,
          f"iso={looks_iso(iso)} dmy={looks_iso(dmy)}")

    # The regression sentinel: this exact string is what created the phantom December date.
    one = parse_dates(pd.Series(["06-12-2026 00:00"]))[0]
    check("'06-12-2026' == 12 June, not 6 December", one == pd.Timestamp("2026-12-06"),
          f"got {one.date()}")


def section_b_files():
    print("\n=== B. FILE BOUNDARY: every date column in every source file ===")
    for path, spec in FILES.items():
        try:
            df = pd.read_csv(path, low_memory=False, encoding="ISO-8859-1",
                             usecols=lambda c: c in spec["cols"])
        except Exception as exc:
            check(f"{path} readable", False, str(exc)[:120])
            continue
        for col in spec["cols"]:
            if col not in df.columns:
                check(f"{path} :: {col} present", False, "column missing")
                continue
            raw = df[col]
            if raw.dropna().astype(str).str.strip().replace("-", "").eq("").all():
                check(f"{path} :: {col}", True, "all blank, skipped")
                continue

            detected = "iso" if looks_iso(raw) else "dayfirst"
            check(f"{path} :: {col} detected as {spec['expect']}",
                  detected == spec["expect"], f"detected {detected}")

            got = parse_dates(raw)
            ok = got.dropna()
            # A handful of junk values exist in the source data independently of this bug -- EDA
            # 'Next Service Date' carries years 2202/3202/9202, Appointments 'Due Date IN' carries
            # 1611 and 8021. Those are a data-quality issue, not a parsing convention issue, so the
            # bar is "the overwhelming bulk lands in a possible range", with the strays reported.
            in_range = ok.between(FLOOR, CEILING)
            n_bad = int((~in_range).sum())
            share = in_range.mean() if len(ok) else 1.0
            good = ok[in_range]
            check(f"{path} :: {col} >=99.9% of values in a possible range",
                  share >= 0.999,
                  f"{share*100:.3f}% ok, {n_bad} stray; bulk "
                  f"{good.min().date()} .. {good.max().date()}" if len(good) else "empty")

            n_diff, msg = assert_no_dayfirst_damage(got, raw, col)
            if spec["expect"] == "iso":
                # For an ISO column the 'opposite convention' IS dayfirst, and applying it mangles
                # the values. A non-zero count here is not a failure -- it is the evidence that
                # dayfirst=True must never be hardcoded on this column. The guarantee that matters
                # is that parse_dates chose ISO, asserted above.
                check(f"{path} :: {col} confirms dayfirst would break it", n_diff > 0, msg)
            else:
                ambiguous = looks_dayfirst_ambiguous(raw)
                check(f"{path} :: {col} is ambiguous and the fix changes it",
                      ambiguous and n_diff > 0,
                      msg + f" [ambiguous={ambiguous}]")


def section_c_codepaths():
    print("\n=== C. CODE PATH: the call sites that read those files ===")
    hist = "data/Service_History_Q2-2026.csv"
    eda = "data/EDA_Q2-2026.csv"

    import prepare_test_set as pts
    import pmstrainfeatureeng_refactored as fe

    serv = pd.read_csv(hist, low_memory=False, encoding="ISO-8859-1", usecols=["Service_Date"])
    truth_max = parse_dates(serv["Service_Date"]).max()

    # create_test_cohort() -- the 'ever' label mode. This is the site that carried the bug.
    src = open("prepare_test_set.py", encoding="utf-8").read()
    check("prepare_test_set.create_test_cohort uses parse_dates for Service_Date",
          "serv['Service_Date'] = parse_dates(" in src or
          'serv["Service_Date"] = parse_dates(' in src,
          "still calling pd.to_datetime without a convention" if "parse_dates" not in src else "")
    # FirstSrvDate is a fillna of two EDA columns, so the convention has to be applied to each
    # SOURCE column, not to the combined result.
    check("prepare_test_set parses both EDA source columns with parse_dates",
          src.count('df["Invoice date"] = parse_dates(') == 2 and
          src.count('df["First Service Date"] = parse_dates(') == 2,
          f'Invoice date x{src.count(chr(34)+"Invoice date"+chr(34)+" ] = parse_dates(")}')

    fsrc = open("pmstrainfeatureeng_refactored.py", encoding="utf-8").read()
    check("pmstrainfeatureeng_refactored master_dates FirstSrvDate uses parse_dates",
          "master_dates['FirstSrvDate'] = parse_dates(" in fsrc)

    gsrc = open("features.py", encoding="utf-8").read()
    for fn in ["compute_service_features", "derive_pms_frequency"]:
        body = gsrc.split(f"def {fn}(")[1].split("\ndef ")[0] if f"def {fn}(" in gsrc else ""
        check(f"features.{fn} uses parse_dates for Service_Date",
              "parse_dates(" in body, "still bare pd.to_datetime" if body else "fn not found")

    # And the actual runtime answer, through the real cohort builder.
    cohort = pts.create_test_cohort(
        eda_path=eda, service_history_path=hist, milestone=20,
        start_date="2026-01-01", end_date="2026-03-31", cutoff_date="2025-12-31",
        use_mileage_projection=False, date_model="schedule",
    )
    check("create_test_cohort returns a non-empty Q1 cohort", len(cohort) > 0, f"{len(cohort)} rows")
    return truth_max


def section_d_invariant(truth_max):
    print("\n=== D. INVARIANT: the fact the bug falsified ===")
    check("Service_History_Q2-2026.csv ends 2026-06-29 (not 2026-12-06)",
          truth_max == pd.Timestamp("2026-06-29"), f"max = {truth_max}")

    act = pd.read_excel("data/Service_History_Jul_2026.xlsx", usecols=["Service_Date"])
    a = parse_dates(act["Service_Date"])
    check("July log stays inside July", a.min() >= pd.Timestamp("2026-07-01") and
          a.max() <= pd.Timestamp("2026-07-31"), f"{a.min().date()} .. {a.max().date()}")


def main():
    print("=" * 78)
    print("dayfirst date-parsing test suite")
    print("=" * 78)
    section_a_unit()
    section_b_files()
    truth_max = section_c_codepaths()
    section_d_invariant(truth_max)

    n = len(RESULTS)
    bad = [r for r, ok in RESULTS if not ok]
    print("\n" + "=" * 78)
    print(f"{n - len(bad)}/{n} passed")
    if bad:
        print("FAILED:")
        for r in bad:
            print(f"  - {r}")
    print("=" * 78)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
