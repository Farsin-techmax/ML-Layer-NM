# MI Scores Analysis — 20k Milestone Feature Selection

## Summary

| Metric | Value |
|---|---|
| Total features scored | 177 |
| Features selected (top 85% MI) | **32** |
| Features dropped (zero MI) | 55 |
| Features dropped (low MI tail) | 90 |

> [!TIP]
> The selection is working correctly. 32 features capture 85% of the total predictive signal, while the remaining 145 features contribute only 15% (mostly noise). This is an excellent compression ratio for model performance.

---

## Top 32 Selected Features (Ranked by MI Score)

| Rank | Feature | MI Score | Cumulative % | Category |
|---:|---|---:|---:|---|
| 1 | `Vehicle_Key_Actual_Service` | 0.384 | 9.9% | Service History |
| 2 | `PMS_Delay` | 0.381 | 19.8% | Service Behavior |
| 3 | `Vehicle Lifetime in Months` | 0.259 | 26.5% | Vehicle Age |
| 4 | `Service Frequency` | 0.246 | 32.8% | Service Behavior |
| 5 | `PMSRevenue` | 0.135 | 36.3% | Revenue |
| 6 | `Max_PMS_Revenue` | 0.130 | 39.7% | Revenue |
| 7 | `Last_PMS_Revenue` | 0.121 | 42.8% | Revenue |
| 8 | `Months_Since_Last_PMS` | 0.111 | 45.7% | Recency |
| 9 | `PMS Status_Non PMS` | 0.096 | 48.2% | Service Status |
| 10 | `Vehicle Service Status_Active` | 0.095 | 50.7% | Service Status |
| 11 | `PMS Status_PMS Partial` | 0.091 | 53.0% | Service Status |
| 12 | `Vehicle Service Status_InActive` | 0.079 | 55.1% | Service Status |
| 13 | `Min_PMS_Revenue` | 0.078 | 57.1% | Revenue |
| 14 | `Weight` | 0.075 | 59.0% | Vehicle Spec |
| 15 | `LastNonPMSMileage` | 0.071 | 60.8% | Mileage |
| 16 | `Service Contract Status_No SC` | 0.070 | 62.6% | Contract |
| 17 | `Avg_Service_Interval_PMS` | 0.067 | 64.4% | Service Intervals |
| 18 | `Service Contract Status_Active` | 0.066 | 66.1% | Contract |
| 19 | `Avg_Mileage_Interval_PMS` | 0.061 | 67.7% | Mileage Intervals |
| 20 | `Avg_Service_Interval_PMSper10k` | 0.058 | 69.2% | Service Intervals |
| 21 | `Contract_PurchaseStatus_Final_No SC` | 0.057 | 70.6% | Contract |
| 22 | `CC` | 0.057 | 72.1% | Vehicle Spec |
| 23 | `21 NM - Alain` | 0.056 | 73.6% | Branch |
| 24 | `Last Service Mileage` | 0.055 | 75.0% | Mileage |
| 25 | `StdDev_PMS_Revenue` | 0.054 | 76.4% | Revenue |
| 26 | `Wheel Base` | 0.052 | 77.2% | Vehicle Spec |
| 27 | `Variant_Cluster_2` | 0.050 | 79.0% | Clustering |
| 28 | `Height` | 0.049 | 80.3% | Vehicle Spec |
| 29 | `Multiplier` | 0.046 | 81.5% | Revenue |
| 30 | `11 NM Airport Road` | 0.045 | 82.6% | Branch |
| 31 | `Variant_Cluster_3` | 0.042 | 83.7% | Clustering |
| 32 | `Avg_Mileage_Interval_NPMS` | 0.035 | 84.7% | Mileage Intervals |

---

## Verification: Does This Make Business Sense?

### ✅ Strong Signals (Top 8 features = 46% of total MI)

The top 8 features alone capture nearly **half** of all predictive power. They tell a clear story:

1. **`Vehicle_Key_Actual_Service`** (MI: 0.384) — How many services a vehicle has actually completed. Directly reflects engagement level. Makes perfect sense as #1.

2. **`PMS_Delay`** (MI: 0.381) — How overdue the vehicle is. A customer who is already 6 months late for a service is far less likely to show up. This is the single strongest behavioral predictor.

3. **`Vehicle Lifetime in Months`** (MI: 0.259) — Older vehicles have different service patterns. New car owners are more compliant; 5+ year owners tend to drift.

4. **`Service Frequency`** (MI: 0.246) — Past behavior predicts future behavior. Frequent servicers are likely to continue.

5-7. **Revenue features** (`PMSRevenue`, `Max_PMS_Revenue`, `Last_PMS_Revenue`) — Customers who spend more on PMS are more invested in the dealership relationship. Strong retention signal.

8. **`Months_Since_Last_PMS`** (MI: 0.111) — Recency signal. The longer since last visit, the less likely they return.

> [!IMPORTANT]
> These top 8 features align perfectly with RFM (Recency, Frequency, Monetary) marketing theory, which is the gold standard for customer retention modeling.

### ✅ Service Status & Contract Features (Ranks 9-21)

The middle tier is dominated by **service status flags** and **service contract status** — whether a customer has an active service contract is a strong predictor of whether they'll show up. This is business-logical: contracted customers have pre-paid and are incentivized to return.

### ✅ Vehicle Specification Features (Ranks 14, 22, 26, 28)

`Weight`, `CC`, `Wheel Base`, `Height` — these are proxies for vehicle segment (luxury SUV vs economy sedan). Different segments have different service compliance rates. The model is correctly learning that vehicle type matters.

### ✅ Branch Features (Ranks 23, 30)

`21 NM - Alain` and `11 NM Airport Road` — specific branches showing up means there are location-specific service patterns. This is valuable geographic signal.

### ✅ Clustering Features (Ranks 27, 31)

`Variant_Cluster_2`, `Variant_Cluster_3` — the hierarchical Gower clustering is working. These clusters capture variant-level behavioral patterns that individual variant names can't.

---

## Correctly Dropped Features

### 55 features with MI = 0.0 (zero predictive power)

These features have absolutely no statistical relationship with whether a customer shows up:

- **All `last_appointment_status_*` OHE flags** — This is interesting. It means appointment status alone doesn't predict PMS compliance for 20k.
- **Most `DeferredPart__*` and `LostPart__*`** — Individual VHC part categories are too sparse to be predictive at this milestone.
- **`Franchise_RENAULT`**, **`Franchise_0`** — Too few vehicles in these franchise categories.
- **`AMA Auto App Sessions`** — App usage has zero signal for 20k (possibly because it's all zeros or constant).

> [!NOTE]
> These zero-MI features being dropped is exactly correct behavior. Keeping them would add noise and hurt model generalization.

---

## Potential Concerns

> [!WARNING]
> **`Vehicle_Key_Actual_Service` at Rank #1 (MI: 0.384):** This feature name is suspicious. If it represents the count of services *including* the target milestone, it could be leaking future information. You mentioned wanting to hard-drop this column. The `feature_sel` function already has it in the `hard_drop` list, but it's appearing in the output — which means **the column name in the DataFrame doesn't exactly match `"Actual Services"`**. The actual column is named `Vehicle_Key_Actual_Service` (after the OHE/merge renaming). You should verify this isn't a data leakage issue.

> [!NOTE]
> **`Vehicle Lifetime in Months` at Rank #3:** You originally wanted to drop `Vehicle Age` and `Purchase Age`. This feature wasn't in your hard-drop list — confirm whether `Vehicle Lifetime in Months` is the same concept or a distinct engineered feature.
