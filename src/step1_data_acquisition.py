"""
Step 1 - Data acquisition
Looks for each series CSV in the data/ folder, downloads any that are missing
(EIA API v2 and FRED API), then merges everything into one weekly file:
data/crude_weekly.csv  (weeks ending Friday, from 1991)
"""
import os
from pathlib import Path

import pandas as pd
import requests

EIA_API_KEY = os.getenv("EIA_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")
DATA_DIR = Path(__file__).parent / "data"
OUTPUT = DATA_DIR / "crude_weekly.csv"
START = "1991-01-01"

EIA_SERIES = {  # file name: PET.WCESTUS1.W -> data/PET.WCESTUS1_W.csv
    # --- Target ---
    "PET.WCESTUS1.W": "crude_stocks_ex_spr",          # Thousand barrels
    # --- Supply ---
    "PET.WCRFPUS2.W": "crude_production",             # Thousand bbl/day
    "PET.WCRIMUS2.W": "crude_imports",                # Thousand bbl/day
    "PET.WCREXUS2.W": "crude_exports",                # Thousand bbl/day
    # --- Demand ---
    "PET.WCRRIUS2.W": "refinery_net_input",           # Thousand bbl/day
    "PET.WPULEUS3.W": "refinery_utilisation",         # Percent
    # --- Inventory context ---
    "PET.W_EPC0_SAX_YCUOK_MBBL.W": "cushing_stocks",  # Thousand barrels
    "PET.WCSSTUS1.W": "spr_stocks",                   # Thousand barrels
    "PET.WGTSTUS1.W": "gasoline_stocks",              # Thousand barrels
    "PET.WDISTUS1.W": "distillate_stocks",            # Thousand barrels
    # --- Prices ---
    "PET.RWTC.W": "wti_price",                        # USD/barrel
    "PET.RBRTE.W": "brent_price",                     # USD/barrel
}

FRED_SERIES = {  # file name: DTWEXBGS -> data/FRED_DTWEXBGS.csv
    "DTWEXBGS": "usd_index",  # Nominal broad dollar index, Jan 2006 = 100 (daily, starts 2006)
}


def download_eia(series_id):
    """Download one weekly series from the EIA API v2 as a date/value table."""
    if not EIA_API_KEY:
        raise SystemExit("Set the EIA_API_KEY environment variable first.")
    url = f"https://api.eia.gov/v2/seriesid/{series_id}"
    params = {"api_key": EIA_API_KEY, "start": START, "length": 5000}
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    rows = response.json()["response"]["data"]
    return pd.DataFrame(rows)[["period", "value"]].rename(columns={"period": "date"})


def download_fred(series_id):
    """Download one daily series from the FRED API as a date/value table."""
    if not FRED_API_KEY:
        raise SystemExit("Set the FRED_API_KEY environment variable first.")
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": series_id, "api_key": FRED_API_KEY,
              "file_type": "json", "observation_start": START}
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    return pd.DataFrame(response.json()["observations"])[["date", "value"]]


def read_series(path, name, download):
    """Read a date/value CSV, downloading it first if the file doesn't exist."""
    if path.exists():
        print(f"found        {path.name}")
    else:
        print(f"downloading  {path.name}")
        download().to_csv(path, index=False)

    df = pd.read_csv(path)  # expects: first column = date, second = value
    values = pd.to_numeric(df.iloc[:, 1], errors="coerce")  # FRED missing days -> NaN
    return pd.Series(values.values, index=pd.to_datetime(df.iloc[:, 0]), name=name).sort_index()


def load_eia(series_id, name):
    file_name = "_".join(series_id.rsplit(".", 1))  # PET.WCESTUS1.W -> PET.WCESTUS1_W
    s = read_series(DATA_DIR / f"{file_name}.csv", name, lambda: download_eia(series_id))
    s.index = s.index + pd.to_timedelta((4 - s.index.weekday) % 7, unit="D")  # week-ending Friday
    return s[~s.index.duplicated()]


def load_fred(series_id, name):
    s = read_series(DATA_DIR / f"FRED_{series_id}.csv", name, lambda: download_fred(series_id))
    return s.resample("W-FRI").mean()  # daily -> weekly average, week ending Friday


def main():
    DATA_DIR.mkdir(exist_ok=True)
    eia = pd.concat([load_eia(sid, name) for sid, name in EIA_SERIES.items()], axis=1)
    fred = pd.concat([load_fred(sid, name) for sid, name in FRED_SERIES.items()], axis=1)

    merged = eia.join(fred)  # keep only the EIA weeks
    merged = merged[merged.index >= START]
    merged.index.name = "date"
    merged.to_csv(OUTPUT)
    print(f"\nSaved {OUTPUT}: {len(merged)} weeks x {merged.shape[1]} series "
          f"({merged.index.min().date()} to {merged.index.max().date()})")


if __name__ == "__main__":
    main()