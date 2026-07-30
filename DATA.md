# Data file vintages (`data/`)

`data/` is gitignored — this file documents what's in it since the code can't.

Two snapshots exist side by side. Same-name `.xlsx`/`.csv` pairs are the same data, different format.

## Legacy snapshot (used by legacy/monolith code)
- `EDA - Q3 2025.csv` / `EDA Datasheet Till 2025.csv`
- `Service History Feb 2026.csv` / `Service History Q1 - 2026.csv`
- `Appoinments - Q3 - 2025.csv` / `Appoinments2025.csv`
- `Digital sessions Q3 - 2025.csv` / `DigitalData2025.csv`
- `RFM Segments Q2 - 2025.csv` / `RFM till 2025.csv`
- `VHC PMS till 2025.csv` / `VHC Q3 - 2025.csv`

## New snapshot (meant for the refactored pipeline)
- `EDA_Q2-2026.csv`
- `Service_History_Q2-2026.csv`
- `Appointments_Q2-2026.csv`
- `Digital_Sessions_Q2-2026.csv`
- `RFM_Q2-2026.csv`
- `Service Code Desc.csv`
- `VHC_Q2-2026.csv`

 

## Known mismatch — not yet resolved (found 2026-07-24)
Scripts in the refactored pipeline currently mix vintages instead of using the new snapshot
throughout:

- `pmstrainfeatureeng_refactored.py` hardcodes `data/EDA - Q3 2025.csv` and
  `data/Service History Q1 - 2026.csv` (both LEGACY), while its RFM/Appointments/Digital/VHC
  paths point at the NEW Q2-2026 files.
- `milestone_features.py` hardcodes `data/Service_History_Q2-2026.csv` (new) but
  `data/EDA Datasheet Till 2025.csv` (legacy) for `EDA_MASTER`.

Before trusting a feature-eng run's output, check which vintage each hardcoded path in the
script actually resolves to — don't assume "refactored script = all new files".
