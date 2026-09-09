"""
01 - Build the modelling dataset: parse responses, join daily meteorological drivers,
and accumulate thermal time (DVI proxy).
Input  : simulated.rds, met_data_calibracao.csv (same folder)
Output : csv/model_data_v2.csv
"""
import os, numpy as np, pandas as pd, pyreadr
from datetime import date, timedelta
BASE = os.path.dirname(os.path.abspath(__file__)); os.makedirs(f"{BASE}/csv", exist_ok=True)

# --- read biometric outputs; parse response columns as numeric ---
b = list(pyreadr.read_r(f"{BASE}/simulated.rds").values())[0]
for c in ['sdm','ldm','gdm','lai','ch']:
    b[c] = pd.to_numeric(b[c].astype(str), errors='coerce')
for c in ['das','doy','year']:
    b[c] = pd.to_numeric(b[c])
b['county'] = b['county'].astype(str).str.strip()

# --- real calendar date of each day of the cycle (from 'das') ---
b = b.sort_values(['county','cultivar','year','treatment','das']).reset_index(drop=True)
grp = ['county','cultivar','year','treatment']
st = b.loc[b.groupby(grp)['das'].idxmin(), grp+['doy','das']].copy()
st['start_date'] = [date(int(y),1,1)+timedelta(days=int(d)-1) for y,d in zip(st['year'], st['doy'])]
st = st.rename(columns={'das':'das0'})
b = b.merge(st[grp+['start_date','das0']], on=grp, how='left')
b['cal_date'] = [pd.Timestamp(s)+pd.Timedelta(days=int(x-x0)) for s,x,x0 in zip(b['start_date'], b['das'], b['das0'])]

# --- sub-daily met -> daily aggregates per site+date ---
m = pd.read_csv(f"{BASE}/met_data_calibracao.csv"); m.columns=[c.strip() for c in m.columns]
m['County']=m['County'].astype(str).str.strip()
m['dt']=pd.to_datetime(m['date_time_local'], format='mixed', errors='coerce')
m['cal_date']=m['dt'].dt.normalize(); m['tC']=m['t']-273.15
daily = m.groupby(['County','cal_date']).agg(
    tmean=('tC','mean'), tmax=('tC','max'), tmin=('tC','min'),
    srad=('sw_down','mean'), diff_rad=('diff_rad','mean'),
    precip=('precip','sum'), q=('q','mean'), wind=('wind','mean')).reset_index().rename(columns={'County':'county'})

# --- join daily drivers + accumulated thermal time (DVI proxy) ---
j = b.merge(daily, on=['county','cal_date'], how='left')
j['gdd_day'] = np.clip((j['tmax']+j['tmin'])/2 - 8.0, 0, None)   # base temperature 8 C
j['gdd_cum'] = j.groupby(grp)['gdd_day'].cumsum()
j = j.dropna(subset=['sdm','ldm','gdm','tmean']).reset_index(drop=True)
j.to_csv(f"{BASE}/csv/model_data_v2.csv", index=False)
print(f"OK: {len(j)} linhas, {j[grp].drop_duplicates().shape[0]} ciclos, anos {sorted(j['year'].unique())}")
print("ranges (t/ha):", {c:(round(j[c].min(),2),round(j[c].max(),2)) for c in ['sdm','ldm','gdm']})
