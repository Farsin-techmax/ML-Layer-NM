"""date_utils.py -- one safe way to turn a date column into datetimes.

WHY THIS EXISTS
The raw data files are day-first:
    EDA_Q2-2026.csv         'Invoice date' etc.  ->  '12/11/2024'      (12 Nov 2024)
    Service_History_*.csv   'Service_Date'       ->  '29-06-2026 00:00'(29 Jun 2026)
    VHC_Q2-2026.csv         'Service Date'       ->  '15/06/2026'      (15 Jun 2026)
but Appointments_Q2-2026.csv is ISO ('2026-06-30', '2026-08-09 00:00:00'), and any frame that has
been through `to_csv` writes its datetimes back out as ISO too.

Parsed without `dayfirst=True`, a day-first value whose day is 1-12 silently becomes a different
real date -- '06-12-2026' reads as 6 Dec instead of 12 Jun. No error, no NaT, just a wrong date.
That is how `Service_History_Q2-2026.csv` came to be documented as running to 2026-12-06 when it
actually ends 2026-06-29.

Parsed WITH `dayfirst=True`, an ISO value with a time component breaks the same way: on pandas
3.0.3, `pd.to_datetime('2026-08-09 00:00:00', format='mixed', dayfirst=True)` returns 2026-09-08.
So `dayfirst=True` is not a safe blanket default either -- it just moves the bug.

`parse_dates()` picks per column by looking at the values, which is the only thing that actually
knows. Use it anywhere a date column might arrive as raw text.
"""
import re

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

# 'YYYY-MM-DD...' -- unambiguous, and dayfirst must NOT be applied to it.
_ISO_PREFIX = re.compile(r"^\s*\d{4}-\d{1,2}-\d{1,2}")
# 'DD/MM/YYYY' or 'DD-MM-YYYY', optionally followed by a time.
_DMY_PREFIX = re.compile(r"^\s*\d{1,2}[/-]\d{1,2}[/-]\d{4}")

_SAMPLE = 2000


def _sample(s, n=_SAMPLE):
    """Non-null, non-'-' values, spread across the whole column rather than taken from the top.

    These files are sorted by date, so `.head()` sees one narrow slice -- a Service_Date column
    whose first thousand rows all fall on days 13-31 looks unambiguous when it is not.
    """
    v = s.dropna().astype(str).str.strip()
    v = v[(v != "-") & (v != "")]
    if len(v) <= n:
        return v
    return v.iloc[:: max(len(v) // n, 1)]


def looks_iso(s, sample=_SAMPLE):
    """True when the column's text values are ISO-ordered (year first)."""
    v = _sample(s, sample)
    if v.empty:
        return False
    return bool(v.str.match(_ISO_PREFIX).mean() > 0.9)


def looks_dayfirst_ambiguous(s, sample=_SAMPLE):
    """True when the column is D/M/Y-shaped AND at least one value could be read either way.

    Used by the tests and by `assert_no_dayfirst_damage` -- a D/M/Y column whose day part is always
    13+ cannot be misparsed, so flagging it would be noise.
    """
    v = _sample(s, sample)
    if v.empty:
        return False
    dmy = v.str.match(_DMY_PREFIX)
    if not bool(dmy.mean() > 0.9):
        return False
    first = v[dmy].str.extract(r"^\s*(\d{1,2})")[0].astype(int)
    return bool((first <= 12).any())


def parse_dates(s, errors="coerce"):
    """Datetime-ify a column, choosing day-first vs ISO from the values themselves.

    - already datetime64  -> returned unchanged (re-parsing is a no-op, but this skips the work)
    - ISO-looking text    -> parsed WITHOUT dayfirst
    - anything else       -> parsed WITH dayfirst (the raw-file convention here)

    '-' is treated as missing, which is how EDA spells a blank date.
    """
    if is_datetime64_any_dtype(s):
        return s
    s = s.replace("-", pd.NA)
    return pd.to_datetime(s, format="mixed", dayfirst=not looks_iso(s), errors=errors)


def assert_no_dayfirst_damage(parsed, raw, name="column"):
    """Cross-check a parsed column against a deliberately wrong parse of the same raw values.

    Returns (n_differing, message). n_differing > 0 on an ambiguous day-first column is EXPECTED and
    is the proof the fix is doing something; on an ISO column it must be 0.
    """
    if is_datetime64_any_dtype(raw):
        return 0, f"{name}: already datetime, nothing to check"
    wrong = pd.to_datetime(raw.replace("-", pd.NA), format="mixed",
                           dayfirst=looks_iso(raw), errors="coerce")
    diff = int((parsed.notna() & wrong.notna() & (parsed != wrong)).sum())
    return diff, f"{name}: {diff:,} values differ from the opposite-convention parse"
