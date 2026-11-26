"""Features v2 for the D' retrain: ONE code path for every row, everything cut at the row's cutoff.
Nothing here reads TargetFlag, Actual{m}kDate or any outcome column -- test_no_label_leak.py checks that.

Sources (decisions 2026-10-15/16):
  service history  data/Service_History_Q2-2026.csv, rows dated <= cutoff, PMS rows >= m excluded
  VHC              data/v2/VHC_Q3-2026.csv, a record counts only when BOTH 'Service Date' and
                   'VHC Completed' are <= cutoff; joined to VIN via Vehicle Key
  appointments     data/v2/Service_Appointments_Q3-2026.csv, booked <= cutoff; show / no-show only
                   for appointments DUE <= cutoff (the outcome of a later one is not known yet);
                   extract-relative flags (Future Booking, Soon To No Show) are not used
  EDA              data/EDA_Q2-2026.csv, static facts only (franchise, model, new/used, AMA, gender,
                   nationality, engine/body specs, invoice date)
  NOT used         RFM (snapshot), digital sessions (no per-record date), every other EDA field

v3 (block_features(..., behaviour=True, prior=prior_listing_table(lists))) adds customer behaviour, all
from history <= cutoff: prior listings for the same milestone (list MEMBERSHIP of earlier quarters only
-- a car still due now did not come in any of them, so no outcome column is read), punctuality vs the
6-month schedule and vs the milestone km, gap regularity / slowing down, rhythm at the cutoff, season,
and the same customer's other vehicles (EDA Customer ID; only their visits <= cutoff are counted).
"""
import numpy as np
import pandas as pd

from date_utils import parse_dates
from features import resolve_service_num

TOP_BRANCHES = 8
TOP_MODELS = 15
TOP_NATIONALITY = 8


class Sources:
    """Load and normalise every table once; feature blocks then only filter."""

    def __init__(self, eda, serv):
        self.serv = serv  # already parsed, Service_Num resolved, Mileage numeric
        self.eda = eda.drop_duplicates("VIN").set_index("VIN")
        key_map = pd.concat([
            serv[["Vehicle_Key", "Vin_No"]].rename(columns={"Vehicle_Key": "key", "Vin_No": "VIN"}),
            eda[["Vehicle Key", "VIN"]].rename(columns={"Vehicle Key": "key"}),
        ]).dropna().astype(str).drop_duplicates("key").set_index("key")["VIN"]
        magic_map = (serv[["Vehicle Magic", "Vin_No"]].dropna().astype(str)
                     .drop_duplicates("Vehicle Magic").set_index("Vehicle Magic")["Vin_No"])

        v = pd.read_csv("data/v2/VHC_Q3-2026.csv", low_memory=False, encoding="ISO-8859-1")
        v["VIN"] = v["Vehicle Key"].astype(str).map(key_map)
        sd, vc = parse_dates(v["Service Date"]), parse_dates(v["VHC Completed"])
        v["known_date"] = pd.concat([sd, vc], axis=1).max(axis=1)
        v["VHC Revenue"] = pd.to_numeric(v["VHC Revenue"], errors="coerce").fillna(0)
        self.vhc = v.dropna(subset=["VIN", "known_date"])

        a = pd.read_csv("data/v2/Service_Appointments_Q3-2026.csv", low_memory=False, encoding="ISO-8859-1")
        a["VIN"] = a["Vehicle Magic"].astype(str).map(magic_map)
        a["booked"] = parse_dates(a["WIP Booking Date"])
        a["due"] = parse_dates(a["Due Date IN"].replace("-", np.nan))
        a.loc[a["due"] > pd.Timestamp("2027-12-31"), "due"] = pd.NaT
        a["showed"] = pd.to_numeric(a["[Vehicle Checkedin (Show Flag)]"], errors="coerce").fillna(0)
        a["noshow"] = pd.to_numeric(a["No Show"], errors="coerce").fillna(0)
        self.appt = a.dropna(subset=["VIN", "booked"])

        b = serv["Service_Branch_Name"].astype(str)
        self.top_branches = [x for x in b.value_counts().index if x != "-"][:TOP_BRANCHES]
        self.top_models = self.eda["Model"].astype(str).value_counts().index[:TOP_MODELS].tolist()
        self.top_nat = [x for x in self.eda["Nationality"].astype(str).value_counts().index
                        if x not in ("Unknown", "nan")][:TOP_NATIONALITY]
        self.static = self._static()

    def _static(self):
        e = self.eda
        s = pd.DataFrame(index=e.index)
        for c in ["CC", "Number of Cylinders", "Weight", "Height", "Wheel Base"]:
            s[c.lower().replace(" ", "_")] = pd.to_numeric(e[c], errors="coerce")
        fr = e["Franchise"].astype(str)
        for x in ["NISSAN", "INFINITI", "RENAULT"]:
            s[f"franchise_{x.lower()}"] = (fr == x).astype(int)
        s["is_used"] = (e["New / Used"].astype(str) == "USED").astype(int)
        s["is_ama"] = (e["AMA or Non AMA"].astype(str) == "AMA Vehicle").astype(int)
        g = e["Gender"].astype(str)
        s["gender_male"], s["gender_female"] = (g == "Male").astype(int), (g == "Female").astype(int)
        mo = e["Model"].astype(str)
        for x in self.top_models:
            s[f"model_{x.lower().replace(' ', '_')}"] = (mo == x).astype(int)
        na = e["Nationality"].astype(str)
        for x in self.top_nat:
            s[f"nat_{x.lower().replace(' ', '_')}"] = (na == x).astype(int)
        inv = parse_dates(e["Invoice date"])
        s["has_invoice"] = inv.notna().astype(int)
        s["_invoice_date"] = inv
        return s


def load_sources():
    """The one loading path for build_matrix.py and test_no_label_leak.py."""
    from pmstrainfeatureeng_refactored import _apply_first_srv_date
    serv = pd.read_csv("data/Service_History_Q2-2026.csv", low_memory=False, encoding="ISO-8859-1")
    serv["Service_Date"] = parse_dates(serv["Service_Date"])
    serv["Mileage"] = pd.to_numeric(serv["Mileage"], errors="coerce").where(lambda x: x <= 500_000)
    serv["Revenue"] = pd.to_numeric(serv["Revenue"], errors="coerce").fillna(0)
    serv = serv.dropna(subset=["Service_Date"])
    serv["Service_Num"] = resolve_service_num(serv)
    eda = pd.read_csv("data/EDA_Q2-2026.csv", low_memory=False, encoding="ISO-8859-1")
    eda = eda.query("`Last Service - PMS` != '-'").copy()
    for c in ["Invoice date", "First Service Date", "Last Service Date"]:
        eda[c] = parse_dates(eda[c])
    eda = _apply_first_srv_date(eda, serv)
    return Sources(eda, serv)


def quarter_index(cq):
    """'2024Q3' -> 2024*4 + 2, so consecutive quarters differ by 1."""
    return int(cq[:4]) * 4 + int(cq[5]) - 1


def prior_listing_table(lists_all):
    """v3: list MEMBERSHIP per (VIN, quarter) from lists_{m}k.csv -- only VIN and CohortQuarter are read.
    For a car that is still due now (not an early completer), every earlier listing for the same
    milestone was by construction a quarter it did not come in, so no outcome column is needed."""
    t = lists_all[["VIN", "CohortQuarter"]].drop_duplicates().copy()
    t["qi"] = t["CohortQuarter"].map(quarter_index)
    return t[["VIN", "qi"]]


def prior_listing_features(table, V, cq):
    qi = quarter_index(cq)
    before = table[table["VIN"].isin(V) & (table["qi"] < qi)]
    g = before.groupby("VIN")["qi"]
    P = pd.DataFrame(index=V)
    P["prior_listings"] = g.size().reindex(V).fillna(0)
    P["listed_prev_quarter"] = before[before["qi"] == qi - 1].groupby("VIN").size().reindex(V).notna().astype(int)
    P["quarters_since_first_listed"] = (qi - g.min()).reindex(V)
    return P


def behaviour_features(src, V, m, cutoff, qs, qe, prior):
    """v3 additions -- customer behaviour, all from history dated <= cutoff (no outcome columns):
    punctuality vs the 6-month schedule and vs the milestone km, gap regularity / slowing down,
    rhythm at the cutoff, season, the customer's other vehicles, and the prior-listing counts."""
    F = prior_listing_features(prior, V, f"{qs.year}Q{(qs.month - 1) // 3 + 1}") if prior is not None \
        else pd.DataFrame(index=V)
    s = src.serv
    S = s[s["Vin_No"].isin(V) & (s["Service_Date"] <= cutoff) & ~(s["Service_Num"] >= m)]
    vis = S.groupby(["Vin_No", "Service_Date"]).agg(num=("Service_Num", "max"), km=("Mileage", "max")).reset_index()
    first_visit = vis.groupby("Vin_No")["Service_Date"].min()
    pms = vis[vis["num"] > 0].sort_values(["Vin_No", "Service_Date"]).copy()
    pms["gap"] = pms.groupby("Vin_No")["Service_Date"].diff().dt.days
    gs = pms.dropna(subset=["gap"]).groupby("Vin_No")["gap"]
    mean_gap = gs.mean().reindex(V)
    F["gap_std_days"] = gs.std().reindex(V)
    F["gap_cv"] = F["gap_std_days"] / mean_gap.where(mean_gap > 0)
    last_gap = gs.last().reindex(V)
    prior_mean = ((gs.sum() - gs.last()) / (gs.count() - 1)).where(gs.count() >= 2).reindex(V)
    F["last_gap_vs_prior_mean"] = last_gap / prior_mean.where(prior_mean > 0)
    F["last_gap_minus_prior_mean_days"] = last_gap - prior_mean
    last_pms = pms.groupby("Vin_No")["Service_Date"].max().reindex(V)
    F["rhythm_at_cutoff"] = (cutoff - last_pms).dt.days / mean_gap.where(mean_gap > 0)

    steps = pms[pms["num"] >= 10].copy()
    steps["expected"] = steps["Vin_No"].map(first_visit) + pd.to_timedelta(steps["num"] / 10 * 182.62, unit="D")
    steps["sched_late"] = (steps["Service_Date"] - steps["expected"]).dt.days
    steps["km_late"] = (steps["km"] - steps["num"] * 1000).where(steps["km"] > 0)
    st = steps.groupby("Vin_No")
    F["sched_late_mean_days"] = st["sched_late"].mean().reindex(V)
    F["sched_late_last_days"] = st["sched_late"].last().reindex(V)
    F["km_late_mean"] = st["km_late"].mean().reindex(V)
    F["km_late_last"] = steps.dropna(subset=["km_late"]).groupby("Vin_No")["km_late"].last().reindex(V)

    for k in range(1, 5):
        F[f"season_q{k}"] = int((qs.month - 1) // 3 + 1 == k)

    cust = src.eda["Customer ID"].reindex(V)
    pairs = (pd.DataFrame({"VIN": V, "cust": cust.values}).dropna()
             .merge(src.eda["Customer ID"].rename("cust").rename_axis("other").reset_index(), on="cust"))
    pairs = pairs[pairs["VIN"] != pairs["other"]]
    o = s[s["Vin_No"].isin(pairs["other"].unique()) & (s["Service_Date"] <= cutoff)]
    ov = o.groupby(["Vin_No", "Service_Date"]).size().reset_index()[["Vin_No", "Service_Date"]]
    per = ov.groupby("Vin_No").agg(last=("Service_Date", "max"))
    per["n365"] = ov[ov["Service_Date"] > cutoff - pd.Timedelta(days=365)].groupby("Vin_No").size().reindex(per.index).fillna(0)
    pairs = pairs.merge(per, left_on="other", right_index=True, how="inner")
    agg = pairs.groupby("VIN").agg(other_vehicles_seen=("other", "nunique"),
                                   other_visits_365d=("n365", "sum"), other_last=("last", "max"))
    F["other_vehicles_seen"] = agg["other_vehicles_seen"].reindex(V).fillna(0)
    F["other_visits_365d"] = agg["other_visits_365d"].reindex(V).fillna(0)
    F["days_since_other_vehicle_visit"] = (cutoff - agg["other_last"].reindex(V)).dt.days
    return F


def block_features(src, lst, m, cutoff, qs, qe, behaviour=False, prior=None):
    """Features for one (milestone, quarter) list. `lst` is the slice of lists_{m}k.csv for this
    quarter (VIN + leg columns only are used). Returns a frame indexed by VIN.
    behaviour=True adds the v3 behaviour block (prior = prior_listing_table(...) for the milestone)."""
    V = pd.Index(lst["VIN"].unique())
    q_mid = qs + (qe - qs) / 2
    F = pd.DataFrame(index=V)

    # --- list context (how the due-date rule listed the car) ---
    L = lst.set_index("VIN")
    for c in ["by_burn", "by_last6", "by_avg", "n_legs", "gap"]:
        F[c] = L[c].reindex(V).astype(float)
    for c in ["burn_date", "last6_date", "avg_date"]:
        d = pd.to_datetime(L[c].reindex(V))
        F[c.replace("_date", "_days_from_qmid")] = (d - q_mid).dt.days
    F["centre_days"] = F[["burn_days_from_qmid", "last6_days_from_qmid", "avg_days_from_qmid"]].abs().where(
        L[["by_burn", "by_last6", "by_avg"]].reindex(V).astype(bool).values).min(axis=1)

    # --- service history, cut at the cutoff, PMS >= m excluded ---
    s = src.serv
    S = s[s["Vin_No"].isin(V) & (s["Service_Date"] <= cutoff) & ~(s["Service_Num"] >= m)]
    vis = S.groupby(["Vin_No", "Service_Date"]).agg(num=("Service_Num", "max"), rev=("Revenue", "sum"),
                                                    km=("Mileage", "max")).reset_index()
    g = vis.groupby("Vin_No")
    F["visits_total"] = g.size().reindex(V).fillna(0)
    F["visits_365d"] = vis[vis["Service_Date"] > cutoff - pd.Timedelta(days=365)].groupby("Vin_No").size().reindex(V).fillna(0)
    first_visit, last_visit = g["Service_Date"].min().reindex(V), g["Service_Date"].max().reindex(V)
    F["days_since_any_visit"] = (cutoff - last_visit).dt.days
    F["age_days"] = (cutoff - first_visit).dt.days
    inv = src.static["_invoice_date"].reindex(V)
    F["invoice_to_first_visit_days"] = (first_visit - inv).dt.days

    pms = vis[vis["num"] > 0].sort_values(["Vin_No", "Service_Date"])
    gp = pms.groupby("Vin_No")
    F["n_pms"] = gp.size().reindex(V).fillna(0)
    for x in range(10, m, 10):
        F[f"has_{x}"] = S[S["Service_Num"] == x].groupby("Vin_No").size().reindex(V).notna().astype(int)
    last_pms = gp["Service_Date"].max().reindex(V)
    F["days_since_last_pms"] = (cutoff - last_pms).dt.days
    gaps = pms.assign(gap=gp["Service_Date"].diff().dt.days).dropna(subset=["gap"])
    F["pms_gap_mean_days"] = gaps.groupby("Vin_No")["gap"].mean().reindex(V)
    F["pms_gap_last_days"] = gaps.groupby("Vin_No")["gap"].last().reindex(V)
    F["rhythm_qend"] = (qe - last_pms).dt.days / F["pms_gap_mean_days"].where(F["pms_gap_mean_days"] > 0)
    F["pms_rev_mean"] = gp["rev"].mean().reindex(V)

    non = vis[vis["num"] == 0]
    F["n_nonpms"] = non.groupby("Vin_No").size().reindex(V).fillna(0)
    F["days_since_last_nonpms"] = (cutoff - non.groupby("Vin_No")["Service_Date"].max().reindex(V)).dt.days
    lp = non["Vin_No"].map(last_pms)
    F["nonpms_since_last_pms"] = non[lp.isna() | (non["Service_Date"] > lp)].groupby("Vin_No").size().reindex(V).fillna(0)
    F["nonpms_rev_sum"] = non.groupby("Vin_No")["rev"].sum().reindex(V).fillna(0)

    F["rev_total"] = S.groupby("Vin_No")["Revenue"].sum().reindex(V).fillna(0)
    F["rev_365d"] = S[S["Service_Date"] > cutoff - pd.Timedelta(days=365)].groupby("Vin_No")["Revenue"].sum().reindex(V).fillna(0)
    for c in ["Tyre Revenue", "Battery Revenue", "Brake Revenue"]:
        F[c.lower().replace(" ", "_")] = pd.to_numeric(S[c], errors="coerce").groupby(S["Vin_No"]).sum().reindex(V).fillna(0)

    od = vis[vis["km"] > 0].sort_values(["Vin_No", "Service_Date"])
    last_od = od.groupby("Vin_No").last()
    F["last_km"] = last_od["km"].reindex(V)
    last_km_date = last_od["Service_Date"].reindex(V)
    life = (last_km_date - first_visit).dt.days
    F["km_per_day"] = F["last_km"] / life.where(life >= 30)
    od = od.merge(last_od[["Service_Date", "km"]].rename(columns={"Service_Date": "ld", "km": "lk"}),
                  left_on="Vin_No", right_index=True)
    early = od[od["Service_Date"] <= od["ld"] - pd.Timedelta(days=30)].groupby("Vin_No").last()
    span = (early["ld"] - early["Service_Date"]).dt.days
    F["km_per_day_recent"] = ((early["lk"] - early["km"]) / span).where(lambda r: r >= 0).reindex(V)
    F["km_to_target_at_qmid"] = m * 1000 - (F["last_km"] + F["km_per_day"] * (q_mid - last_km_date).dt.days)

    br = S["Service_Branch_Name"].astype(str)
    F["n_branches"] = br.groupby(S["Vin_No"]).nunique().reindex(V).fillna(0)
    main = br.groupby(S["Vin_No"]).agg(lambda x: x.value_counts().index[0]).reindex(V)
    F["share_main_branch"] = br.groupby(S["Vin_No"]).agg(lambda x: x.value_counts().iloc[0] / len(x)).reindex(V)
    for i, b in enumerate(src.top_branches):
        F[f"main_branch_{i + 1}"] = (main == b).astype(int)
    F["bodyshop_visits"] = S[br.str.contains("Bodyshop", case=False)].groupby("Vin_No")["Service_Date"].nunique().reindex(V).fillna(0)
    sc = pd.to_numeric(S["Survey Score"], errors="coerce")
    F["survey_mean"] = sc.groupby(S["Vin_No"]).mean().reindex(V)
    F["n_detractor"] = (S["Survey Status"].astype(str) == "Detractor").groupby(S["Vin_No"]).sum().reindex(V).fillna(0)
    F["n_keys"] = S.groupby("Vin_No")["Vehicle_Key"].nunique().reindex(V).fillna(0)

    # --- VHC, known by the cutoff ---
    vh = src.vhc[src.vhc["VIN"].isin(V) & (src.vhc["known_date"] <= cutoff)]
    gv = vh.groupby("VIN")
    F["has_vhc"] = gv.size().reindex(V).notna().astype(int)
    F["vhc_visits"] = gv["WIP No"].nunique().reindex(V).fillna(0)
    F["vhc_red"] = (vh["Criticality"] == "Red").groupby(vh["VIN"]).sum().reindex(V).fillna(0)
    F["vhc_amber"] = (vh["Criticality"] == "Amber").groupby(vh["VIN"]).sum().reindex(V).fillna(0)
    for st in ["Deferred", "Invoiced", "Deleted"]:
        F[f"vhc_{st.lower()}"] = (vh["VHS Status"] == st).groupby(vh["VIN"]).sum().reindex(V).fillna(0)
        F[f"vhc_rev_{st.lower()}"] = vh["VHC Revenue"].where(vh["VHS Status"] == st, 0).groupby(vh["VIN"]).sum().reindex(V).fillna(0)
    F["vhc_deferred_red"] = ((vh["VHS Status"] == "Deferred") & (vh["Criticality"] == "Red")).groupby(vh["VIN"]).sum().reindex(V).fillna(0)
    F["days_since_vhc"] = (cutoff - gv["known_date"].max().reindex(V)).dt.days

    # --- appointments: booked by the cutoff; outcomes only for those due by the cutoff ---
    ap = src.appt[src.appt["VIN"].isin(V) & (src.appt["booked"] <= cutoff)]
    past = ap[ap["due"] <= cutoff]
    F["has_appt"] = ap.groupby("VIN").size().reindex(V).notna().astype(int)
    F["appt_booked"] = ap.groupby("VIN").size().reindex(V).fillna(0)
    F["appt_showed"] = past.groupby("VIN")["showed"].sum().reindex(V).fillna(0)
    F["appt_noshow"] = past.groupby("VIN")["noshow"].sum().reindex(V).fillna(0)
    F["appt_noshow_rate"] = F["appt_noshow"] / past.groupby("VIN").size().reindex(V)
    F["days_since_last_booking"] = (cutoff - ap.groupby("VIN")["booked"].max().reindex(V)).dt.days
    up = ap[ap["due"] > cutoff]
    F["upcoming_appt_in_quarter"] = up[up["due"] <= qe].groupby("VIN").size().reindex(V).notna().astype(int)
    F["upcoming_appt_later"] = up[up["due"] > qe].groupby("VIN").size().reindex(V).notna().astype(int)

    # --- static EDA facts ---
    st = src.static.drop(columns="_invoice_date").reindex(V)
    out = pd.concat([F, st], axis=1)
    if behaviour:
        out = pd.concat([out, behaviour_features(src, V, m, cutoff, qs, qe, prior)], axis=1)
    return out
