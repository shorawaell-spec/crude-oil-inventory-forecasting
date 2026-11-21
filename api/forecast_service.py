"""
forecast_service.py - live forecast for the website (Tech Fem, SIC capstone)

One call does everything, straight from the EIA API, with no CSV/parquet files
and no notebook re-runs:

    fetch 12 EIA series -> build the same features as step 2
    -> refit LightGBM on ALL labelled weeks (incl. the 2 test years)
    -> forecast next week's stock change -> JSON

How the target is "generated": the label of week t is the stock change of week
t+1. Every time EIA publishes a new week, last week's label becomes known
automatically, and the newest week is the one row without a label - that is the
week we forecast. Nothing has to be created by hand.

Usage
    export EIA_API_KEY=...
    python forecast_service.py             # prints + saves forecast_latest.json
    from forecast_service import run_forecast; result = run_forecast()   # in a web backend
"""
import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import lightgbm as lgb

START = "1991-01-01"
OUTPUT_FILE = os.getenv("FORECAST_OUTPUT", "forecast_latest.json")
TARGET = "crude_stocks_ex_spr"
LABEL = "target_next_week_change"

EIA_SERIES = {
    "PET.WCESTUS1.W": "crude_stocks_ex_spr",
    "PET.WCRFPUS2.W": "crude_production",
    "PET.WCRIMUS2.W": "crude_imports",
    "PET.WCREXUS2.W": "crude_exports",
    "PET.WCRRIUS2.W": "refinery_net_input",     # only used inside implied_change
    "PET.WPULEUS3.W": "refinery_utilisation",
    "PET.W_EPC0_SAX_YCUOK_MBBL.W": "cushing_stocks",
    "PET.WCSSTUS1.W": "spr_stocks",
    "PET.WGTSTUS1.W": "gasoline_stocks",
    "PET.WDISTUS1.W": "distillate_stocks",
    "PET.RWTC.W": "wti_price",
    "PET.RBRTE.W": "brent_price",               # only used inside brent_wti_spread
}

# Must be exactly the columns the evaluated model was trained on (same order).
# If the team's final model used a selected subset (ReliefF / mRMR), list it here.
FEATURE_COLS = [
    "crude_stocks_ex_spr", "crude_production", "crude_imports", "crude_exports",
    "refinery_utilisation", "cushing_stocks", "spr_stocks", "gasoline_stocks",
    "distillate_stocks", "wti_price", "stocks_change", "stocks_change_4wk_avg",
    "spr_change", "implied_change", "brent_wti_spread", "stocks_vs_5yr_avg",
    "week_sin", "week_cos", "is_event",
]

EVENTS = {  # same windows as step 2; None = still ongoing
    "Gulf War": ("1991-01-01", "1991-03-31"),
    "Hurricanes Katrina & Rita": ("2005-08-26", "2005-10-31"),
    "Hurricanes Gustav & Ike": ("2008-08-29", "2008-10-10"),
    "2008 financial crisis": ("2008-09-15", "2009-06-30"),
    "Hurricane Harvey": ("2017-08-25", "2017-10-06"),
    "COVID-19 demand collapse": ("2020-03-01", "2020-07-31"),
    "Winter Storm Uri": ("2021-02-12", "2021-03-12"),
    "Hurricane Ida": ("2021-08-27", "2021-09-24"),
    "Russia-Ukraine war & SPR release": ("2022-02-24", "2022-12-31"),
    "Israel-Iran 12-day war": ("2025-06-13", "2025-06-27"),
    "2026 Iran war & Hormuz crisis": ("2026-02-28", None),
}

MODEL_PARAMS = dict(n_estimators=200, learning_rate=0.01, max_depth=3, num_leaves=7,
                    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                    random_state=42, verbose=-1)
HALF_LIFE_WEEKS = 156                   # recency weighting (same as evaluation)
LOWER_OFFSET, UPPER_OFFSET = -5744.5, 5413.6   # 10th / 90th pct of backtest residuals


# -- 1. Data ------------------------------------------------------------------
def fetch_series(series_id, api_key):
    url = f"https://api.eia.gov/v2/seriesid/{series_id}"
    params = {"api_key": api_key, "start": START, "length": 5000}
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    rows = response.json()["response"]["data"]
    s = pd.Series(pd.to_numeric([r["value"] for r in rows], errors="coerce"),
                  index=pd.to_datetime([r["period"] for r in rows]))
    s.index = s.index + pd.to_timedelta((4 - s.index.weekday) % 7, unit="D")  # week-ending Friday
    return s[~s.index.duplicated()].sort_index()


def fetch_raw(api_key=None):
    """Download all 12 series and return one weekly table (same as step 1)."""
    api_key = api_key or os.getenv("EIA_API_KEY")
    if not api_key:
        raise RuntimeError("Set the EIA_API_KEY environment variable.")
    raw = pd.concat({name: fetch_series(sid, api_key) for sid, name in EIA_SERIES.items()}, axis=1)
    return raw[raw.index >= START]


# -- 2. Features (same rules as the step 2 notebook) --------------------------
def build_features(raw):
    df = raw.apply(pd.to_numeric, errors="coerce").sort_index()
    calendar = pd.date_range(df.index.min(), df.index.max(), freq="W-FRI")
    df = df.reindex(calendar).interpolate(method="time", limit_area="inside").ffill(limit=2)
    if df["cushing_stocks"].median() < 1_000:              # same unit guard as step 2
        df["cushing_stocks"] *= 1_000

    fe = df.copy()
    fe["stocks_change"] = fe[TARGET].diff()
    fe[LABEL] = fe["stocks_change"].shift(-1)           # next week's change = the label
    fe["stocks_change_4wk_avg"] = fe["stocks_change"].rolling(4).mean()
    fe["spr_change"] = fe["spr_stocks"].diff()
    flows = fe["crude_production"] + fe["crude_imports"] - fe["crude_exports"] - fe["refinery_net_input"]
    fe["implied_change"] = 7 * flows - fe["spr_change"]
    fe["brent_wti_spread"] = fe["brent_price"] - fe["wti_price"]
    past = pd.concat([fe[TARGET].shift(52 * k) for k in range(1, 6)], axis=1)
    fe["stocks_vs_5yr_avg"] = fe[TARGET] - past.mean(axis=1, skipna=False)
    week = fe.index.isocalendar().week.astype(float).to_numpy()
    fe["week_sin"] = np.sin(2 * np.pi * week / 52.18)
    fe["week_cos"] = np.cos(2 * np.pi * week / 52.18)
    fe["is_event"] = 0
    for start, end in EVENTS.values():
        inside = (fe.index >= pd.Timestamp(start)) & (fe.index <= pd.Timestamp(end or fe.index.max()))
        fe.loc[inside, "is_event"] = 1
    fe.index.name = "date"
    return fe


# -- 3. Refit on all history and forecast -------------------------------------
def train_and_forecast(fe):
    usable = fe.dropna(subset=FEATURE_COLS)              # starts Apr 2004 (Cushing)
    known = usable[usable[LABEL].notna()]                # every week whose next week is published
    latest = usable.iloc[[-1]]                           # newest week: label unknown -> forecast it
    if latest[LABEL].notna().any():
        raise ValueError("Newest week already has a label - the data looks incomplete.")

    y_mean, y_std = known[LABEL].mean(), known[LABEL].std()
    age = np.arange(len(known))[::-1]                     # 0 = most recent week
    weights = np.exp(-np.log(2) / HALF_LIFE_WEEKS * age)

    model = lgb.LGBMRegressor(**MODEL_PARAMS)
    model.fit(known[FEATURE_COLS], (known[LABEL] - y_mean) / y_std, sample_weight=weights)
    point = float(model.predict(latest[FEATURE_COLS])[0]) * y_std + y_mean

    as_of = latest.index[0]
    stock = float(latest[TARGET].iloc[0])
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "model": "LightGBM (refit on full history)",
        "trained_on": f"{known.index[0].date()} to {known.index[-1].date()} ({len(known)} weeks)",
        "as_of_week": as_of.strftime("%Y-%m-%d"),
        "forecast_week": (as_of + timedelta(weeks=1)).strftime("%Y-%m-%d"),
        "current_stock_level_kb": round(stock, 1),
        "point_forecast_kb": round(point, 1),
        "lower_bound_kb": round(point + LOWER_OFFSET, 1),
        "upper_bound_kb": round(point + UPPER_OFFSET, 1),
        "forecast_stock_level_kb": round(stock + point, 1),
        "confidence_interval": "80% (10th-90th percentile of backtest residuals)",
    }


def run_forecast(api_key=None, output_file=OUTPUT_FILE):
    result = train_and_forecast(build_features(fetch_raw(api_key)))
    if output_file:
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    print(json.dumps(run_forecast(), indent=2))
