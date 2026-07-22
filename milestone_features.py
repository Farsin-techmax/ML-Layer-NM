
import numpy as np
import pandas as pd

SERV_HISTORY = "data/Service_History_Q2-2026.csv"
EDA_MASTER = "data/EDA Datasheet Till 2025.csv"

MAX_MONTHS_GAP = 120       # >10 years between two milestones is a data error
MAX_MILEAGE_GAP = 100_000  # >100k km in one service block is a data error


def extract_k(service_str):
    """Service_Num from Description, e.g. 'PMS-30' -> 30. Same rule as the legacy pipeline."""
    try:
        return int(str(service_str).split('-')[-1])
    except Exception:
        return 0


class MilestoneHistory:
    """Loads the raw service history once, then serves per-row features for any frame."""

    def __init__(self, milestone, serv_path=SERV_HISTORY, master_path=EDA_MASTER):
        self.milestone = milestone
        self.prior = [10 * i for i in range(1, milestone // 10)]  # excludes the target milestone

        serv = pd.read_csv(serv_path, low_memory=False, encoding='ISO-8859-1',
                           usecols=['Vin_No', 'Service_Date', 'Description', 'Mileage'])
        serv['Service_Date'] = pd.to_datetime(serv['Service_Date'], format='mixed',
                                              dayfirst=True, errors='coerce')
        serv['Mileage'] = pd.to_numeric(serv['Mileage'], errors='coerce')
        serv['Service_Num'] = serv['Description'].apply(extract_k)
        serv['Service_Num'] = np.where(serv['Description'] == '<=10', 10, serv['Service_Num'])
        serv = serv.dropna(subset=['Service_Date'])

        # one row per (VIN, milestone): the visit that completed it
        self.visits = (serv[serv['Service_Num'].isin(self.prior)]
                       .sort_values('Service_Date')
                       .groupby(['Vin_No', 'Service_Num'], as_index=False)
                       .agg(Service_Date=('Service_Date', 'first'), Mileage=('Mileage', 'first')))

        # start-of-life date per VIN
        master = pd.read_csv(master_path, low_memory=False,
                             usecols=['VIN', 'Invoice date', 'First Service Date'])
        master = master.drop_duplicates(subset='VIN')
        inv = pd.to_datetime(master['Invoice date'], format='mixed', dayfirst=True, errors='coerce')
        fsd = pd.to_datetime(master['First Service Date'], format='mixed', dayfirst=True, errors='coerce')
        master['FirstSrvDate'] = inv.fillna(fsd)
        first_npms = (serv[serv['Service_Num'] < 10].groupby('Vin_No')['Service_Date'].min()
                      .rename('Date_first_npms'))
        base = master.set_index('VIN')['FirstSrvDate'].to_frame().join(first_npms, how='outer')
        self.base_date = base.min(axis=1).rename('Base_Date')

    @property
    def feature_names(self):
        names = ['months_base_to_10'] + [f'months_{p}_to_{c}'
                                         for p, c in zip(self.prior, self.prior[1:])]
        names += ['avg_mileage_between_milestones', 'min_mileage_between_milestones',
                  'max_mileage_between_milestones', 'std_mileage_between_milestones',
                  'n_prior_milestones_done']
        return names

    def features_for(self, vins, cutoffs):
        """Features for each (vin, cutoff) pair. Returns a frame indexed like `vins`."""
        rows = pd.DataFrame({'Vin_No': pd.Series(vins).values,
                             'cutoff': pd.to_datetime(pd.Series(cutoffs).values,
                                                      format='mixed', dayfirst=True, errors='coerce')})
        rows['_row'] = np.arange(len(rows))
        # rows with no usable cutoff fall back to "everything in the history"
        rows['cutoff'] = rows['cutoff'].fillna(pd.Timestamp.max)

        j = rows.merge(self.visits, on='Vin_No', how='left')
        j = j[j['Service_Date'] <= j['cutoff']]

        dates = j.pivot(index='_row', columns='Service_Num', values='Service_Date')
        miles = j.pivot(index='_row', columns='Service_Num', values='Mileage')
        for m in self.prior:                      # guarantee a column per prior milestone
            if m not in dates.columns:
                dates[m], miles[m] = pd.NaT, np.nan
        idx = rows['_row']
        dates = dates[self.prior].reindex(idx)
        miles = miles[self.prior].reindex(idx)

        base = self.base_date.reindex(rows['Vin_No'].values)
        base.index = idx

        out = pd.DataFrame(index=idx)

        # --- family 1: months between consecutive milestones ---
        months = pd.DataFrame(index=idx)
        months['months_base_to_10'] = (dates[10] - base).dt.days / 30.44
        for prev, curr in zip(self.prior, self.prior[1:]):
            months[f'months_{prev}_to_{curr}'] = (dates[curr] - dates[prev]).dt.days / 30.44
        months = months.where((months >= 0) & (months <= MAX_MONTHS_GAP))
        out = out.join(months)

        # --- family 2: mileage driven between consecutive milestones ---
        gaps = pd.DataFrame(index=idx)
        gaps['gap_base_to_10'] = miles[10]                      # from 0 km at base
        for prev, curr in zip(self.prior, self.prior[1:]):
            gaps[f'gap_{prev}_to_{curr}'] = miles[curr] - miles[prev]
        gaps = gaps.where((gaps >= 0) & (gaps <= MAX_MILEAGE_GAP))

        out['avg_mileage_between_milestones'] = gaps.mean(axis=1)
        out['min_mileage_between_milestones'] = gaps.min(axis=1)
        out['max_mileage_between_milestones'] = gaps.max(axis=1)
        out['std_mileage_between_milestones'] = gaps.std(axis=1)
        out['n_prior_milestones_done'] = dates.notna().sum(axis=1)

        out.index = pd.Series(vins).index
        return out


if __name__ == "__main__":
    import sys
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    src = sys.argv[2] if len(sys.argv) > 2 else f"traindataq1_q2/finalmerged{m}kQ1.csv"
    df = pd.read_csv(src, low_memory=False)
    h = MilestoneHistory(m)
    f = h.features_for(df['VIN'], df[f'Next{m}K_Due'])
    print(f.shape)
    print(f.describe().T.to_string())
    print("\ncoverage:", (f.notna().mean().round(3)).to_dict())
