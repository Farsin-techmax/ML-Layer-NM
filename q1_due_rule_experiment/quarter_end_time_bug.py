"""Size of the quarter-end boundary bug: quarter_bounds() ends Q1 at 2026-03-31 00:00, and
burn_rate_date() returns dates with a time of day, so a vehicle due 31 March after midnight fails
`due <= Q_END`. Compares caught completers for rules A/B/C with raw vs day-normalized due dates.
Run from repo root.
"""
import pandas as pd

from build_unionstale_20k import load_base
from due_date import burn_rate_date, last_completed_plus_months, avg_interval_date, earliest_date
from features import expected_milestone_months
from pmstrainfeatureeng_refactored import quarter_bounds

Q_START, Q_END = quarter_bounds(2026, 1)
df, serv = load_base()
vins = df["VIN"].values
first_srv = df.set_index("VIN")["FirstSrvDate"].reindex(vins)
cutoff = serv["Service_Date"].max()
serv_cols = serv[["Vin_No", "Service_Date", "Mileage", "Service_Num"]]


def in_window(due):
    return set(due.index[due.notna() & (due >= Q_START) & (due <= Q_END)])


rows = []
for m in [20, 30, 40, 50]:
    legs = {
        "sched": first_srv + pd.DateOffset(months=expected_milestone_months(m, "schedule")),
        "burn": burn_rate_date(vins, serv_cols, m, cutoff),
        "last": last_completed_plus_months(vins, serv_cols, m, months=6),
        "avg": avg_interval_date(vins, first_srv, serv_cols, m),
    }
    at_ms = serv[serv["Service_Num"] == m].sort_values("Service_Date")
    fm = at_ms.groupby("Vin_No")["Service_Date"].first()
    actual = set(fm[(fm >= Q_START) & (fm <= Q_END)].index)
    for label, L in [("raw", legs), ("day", {k: v.dt.normalize() for k, v in legs.items()})]:
        A = in_window(earliest_date(L["sched"], L["burn"]))
        B = in_window(L["sched"]) | in_window(L["burn"])
        C = B | in_window(L["last"]) | in_window(L["avg"])
        for r, s in [("A", A), ("B", B), ("C", C)]:
            rows.append({"milestone": f"{m}k", "rule": r, "dates": label,
                         "listed": len(s), "caught": len(s & actual)})
t = pd.DataFrame(rows).pivot_table(index=["milestone", "rule"], columns="dates",
                                   values=["listed", "caught"], aggfunc="first")
print(t.to_string())
t.to_csv("q1_due_rule_experiment/quarter_end_time_bug.csv")
