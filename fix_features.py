import sys

with open('features.py', 'r') as f:
    content = f.read()

# Fix 1
content = content.replace(
    '''    servcode_desc = servcode_desc.dropna()
    servcode_desc = dict(zip(servcode_desc['SO_CO_CODE'], servcode_desc['SO_CO_DESCRIPN_001']))''',
    '''    servcode_desc = servcode_desc.dropna()
    sc_col = 'SO_CO_CODE' if 'SO_CO_CODE' in servcode_desc.columns else 'Service_Code'
    servcode_desc = dict(zip(servcode_desc[sc_col], servcode_desc['SO_CO_DESCRIPN_001']))'''
)

# Fix 2
content = content.replace(
    '''    if "Service_Date" in service_df.columns:
        service_df["Service_Date"] = pd.to_datetime(service_df["Service_Date"], errors="coerce", dayfirst=True)
    service_df["Complaint Closed in"] = pd.to_numeric(service_df["Complaint Closed in"], errors="coerce")''',
    '''    if "Service_Date" in service_df.columns:
        service_df["Service_Date"] = pd.to_datetime(service_df["Service_Date"], errors="coerce", dayfirst=True)
    if "Complaint Closed in" in service_df.columns:
        service_df["Complaint Closed in"] = pd.to_numeric(service_df["Complaint Closed in"], errors="coerce")'''
)

content = content.replace(
    '''    for vin, grp in merged.groupby("Vin_No"):
        cats = grp["Complaint Category"].dropna()
        res_vals = grp["Complaint Closed in"].dropna()''',
    '''    for vin, grp in merged.groupby("Vin_No"):
        import pandas as pd
        cats = grp["Complaint Category"].dropna() if "Complaint Category" in grp.columns else pd.Series(dtype=str)
        res_vals = grp["Complaint Closed in"].dropna() if "Complaint Closed in" in grp.columns else pd.Series(dtype=float)'''
)

# Fix 3
content = content.replace(
    '''    # 4. Top Parts OHE
    parts_df = df.dropna(subset=['Refined Description (Enhanced)']).copy()
    parts_df['Refined Description (Enhanced)'] = parts_df['Refined Description (Enhanced)'].astype(str).str.strip()
    
    TOP_K = 20
    for label, col_name in [('Lost', 'LostPart__'), ('Invoiced', 'InvoicedPart__'), ('Deferred', 'DeferredPart__')]:
        sub_df = parts_df[parts_df['VHS Status'] == label]
        if not sub_df.empty:
            top_parts = [p for p, _ in Counter(sub_df['Refined Description (Enhanced)']).most_common(TOP_K)]
            sub_df = sub_df[sub_df['Refined Description (Enhanced)'].isin(top_parts)]
            if not sub_df.empty:
                crosstab = pd.crosstab([sub_df['Vehicle Key'], sub_df['Service Code']], sub_df['Refined Description (Enhanced)']).clip(upper=1)
                crosstab.columns = [f'{col_name}{c}' for c in crosstab.columns]
                grouped = grouped.merge(crosstab.reset_index(), on=['Vehicle Key', 'Service Code'], how='left')''',
    '''    # 4. Top Parts OHE
    desc_col = 'Refined Description (Enhanced)' if 'Refined Description (Enhanced)' in df.columns else 'Refined Description'
    parts_df = df.dropna(subset=[desc_col]).copy()
    parts_df[desc_col] = parts_df[desc_col].astype(str).str.strip()
    
    TOP_K = 20
    for label, col_name in [('Lost', 'LostPart__'), ('Invoiced', 'InvoicedPart__'), ('Deferred', 'DeferredPart__')]:
        sub_df = parts_df[parts_df['VHS Status'] == label]
        if not sub_df.empty:
            top_parts = [p for p, _ in Counter(sub_df[desc_col]).most_common(TOP_K)]
            sub_df = sub_df[sub_df[desc_col].isin(top_parts)]
            if not sub_df.empty:
                crosstab = pd.crosstab([sub_df['Vehicle Key'], sub_df['Service Code']], sub_df[desc_col]).clip(upper=1)
                crosstab.columns = [f'{col_name}{c}' for c in crosstab.columns]
                grouped = grouped.merge(crosstab.reset_index(), on=['Vehicle Key', 'Service Code'], how='left')'''
)

with open('features.py', 'w') as f:
    f.write(content)

print("Done")
