"""
DeshiKrishi data pipeline.

Builds monthly price series for farmer-relevant crops from one of two sources:
  - bbs (default): BBS Monthly Statistical Bulletin, Table 3.1 wholesale prices, 2020-2026
    (data/raw/bbs_wholesale_monthly_clean.csv, built by pipeline/extract_bbs.py + clean_bbs.py)
  - wfp: WFP food prices for Bangladesh (Humanitarian Data Exchange), retail, 2004-2021
It then trains a pooled Gradient Boosting model on month-to-month
price changes, backtests it with rolling-origin evaluation, and writes next-month forecasts.

The Flask app only reads the files this script writes; it never trains models itself.

Usage:
    pip install -r requirements-pipeline.txt
    python pipeline/build_data.py                                   # BBS wholesale, test from 2024-07
    python pipeline/build_data.py --source wfp                      # WFP retail, download from HDX
    python pipeline/build_data.py --source wfp --input wfp_food_prices_bgd.csv

Outputs (in data/):
    prices.csv      monthly price history per series
    backtest.csv    rolling-origin predictions vs actual changes (test period)
    forecasts.csv   next-month forecast per series, with an 80% range
    meta.json       data source, date range, model settings
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HDX_DATASET = "wfp-food-prices-for-bangladesh"
CROPS = ["Rice (coarse)", "Lentils (masur)", "Potatoes", "Onions"]   # farmer-relevant
MIN_MONTHS = 60
TEST_MONTHS = 24
SEED = 42
FEATURES = ["ret_1", "ret_2", "ret_3", "ret_12", "seas_change", "dev_ma6", "month_sin", "month_cos", "crop_code"]

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data"
BBS_CSV = ROOT / "data" / "raw" / "bbs_wholesale_monthly_clean.csv"
BBS_TEST_START = "2024-07-01"


def bbs_series(path=BBS_CSV):
    """BBS clean panel -> the same long format monthly_series() returns."""
    b = pd.read_csv(path, parse_dates=["month"])
    m = pd.DataFrame({"commodity": b["commodity"], "unit": "KG", "market": b["town"],
                      "month": b["month"], "price": b["price_tk_per_kg"]}).dropna(subset=["price"])
    m["price_type"] = "Wholesale"
    m["series_id"] = (m["commodity"] + "|" + m["market"] + "|" + m["unit"]).str.replace(" ", "_")
    return m


def hdx_csv_url(dataset_id=HDX_DATASET):
    import requests
    api = f"https://data.humdata.org/api/3/action/package_show?id={dataset_id}"
    meta = requests.get(api, timeout=60).json()["result"]
    csvs = [r for r in meta["resources"] if r.get("format", "").upper() == "CSV"]
    if not csvs:
        sys.exit("No CSV found in the HDX dataset. Download it by hand and use --input.")
    csvs.sort(key=lambda r: ("food_prices" not in r["url"], "qc" in r["url"]))
    return csvs[0]["url"]


def load_raw(path_or_url):
    df = pd.read_csv(path_or_url, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    df = df[~df["date"].astype(str).str.startswith("#")]          # HXL tag row
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["date", "price"])
    df = df[df["price"] > 0]
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return df


def monthly_series(df):
    """One price per crop, market, unit and month. Prefer wholesale when a crop has enough of it."""
    df = df[df["commodity"].isin(CROPS)]
    parts = []
    for crop, g in df.groupby("commodity"):
        ptype = "Wholesale" if (g["pricetype"] == "Wholesale").sum() >= MIN_MONTHS else "Retail"
        g = g[g["pricetype"] == ptype]
        m = g.groupby(["commodity", "unit", "market", "month"], as_index=False)["price"].mean()
        m["price_type"] = ptype
        parts.append(m)
    if not parts:
        sys.exit(f"None of {CROPS} found in the data.")
    m = pd.concat(parts)
    counts = m.groupby(["commodity", "unit", "market"])["month"].nunique()
    keep = counts[counts >= MIN_MONTHS].index
    m = m.set_index(["commodity", "unit", "market"]).loc[keep].reset_index()
    m["series_id"] = (m["commodity"] + "|" + m["market"] + "|" + m["unit"]).str.replace(" ", "_")
    return m


def features(m):
    crop_codes = {c: i for i, c in enumerate(sorted(m["commodity"].unique()))}
    frames = []
    for sid, g in m.groupby("series_id"):
        s = (g.set_index("month")["price"].sort_index().asfreq("MS")
             .interpolate(limit=2, limit_area="inside"))
        f = pd.DataFrame({"price": s})
        f["target"] = s.shift(-1) / s - 1
        f["ret_1"] = s / s.shift(1) - 1
        f["ret_2"] = s / s.shift(2) - 1
        f["ret_3"] = s / s.shift(3) - 1
        f["ret_12"] = s / s.shift(12) - 1
        f["seas_change"] = s.shift(11) / s.shift(12) - 1
        f["dev_ma6"] = s / s.rolling(6).mean() - 1
        f["month_sin"] = np.sin(2 * np.pi * f.index.month / 12)
        f["month_cos"] = np.cos(2 * np.pi * f.index.month / 12)
        f["crop_code"] = crop_codes[g["commodity"].iloc[0]]
        f["series_id"] = sid
        frames.append(f.reset_index().rename(columns={"index": "month"}))
    return pd.concat(frames, ignore_index=True)


def make_model():
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05,
                                     subsample=0.8, random_state=SEED)


def backtest(feat, test_start=None):
    """Rolling origin: for each test month T, train on rows before T (their targets are known by T).
    Test period: months >= test_start if given, else the last TEST_MONTHS of each series."""
    rows = feat.dropna(subset=FEATURES + ["target"]).copy()
    if test_start is not None:
        rows["is_test"] = rows["month"] >= pd.Timestamp(test_start)
    else:
        rows["is_test"] = False
        for sid, g in rows.groupby("series_id"):
            rows.loc[g.index[-TEST_MONTHS:], "is_test"] = True
    out = []
    for T in sorted(rows.loc[rows["is_test"], "month"].unique()):
        train = rows[rows["month"] < T]
        now = rows[(rows["month"] == T) & rows["is_test"]]
        if len(train) < 100 or now.empty:
            continue
        model = make_model().fit(train[FEATURES], train["target"])
        out.append(pd.DataFrame({"series_id": now["series_id"], "month": now["month"], "price": now["price"],
                                 "pred_return": model.predict(now[FEATURES]),
                                 "actual_return": now["target"]}))
    return pd.concat(out, ignore_index=True)


def forecast(feat, bt):
    """Train on every row with a known target; forecast the month after each series' last price."""
    train = feat.dropna(subset=FEATURES + ["target"])
    model = make_model().fit(train[FEATURES], train["target"])
    last = feat.dropna(subset=FEATURES).sort_values("month").groupby("series_id").tail(1).copy()
    last["pred_return"] = model.predict(last[FEATURES])

    # 80% range from backtest errors of the same crop (pooled, because one series has only ~24 errors)
    crop_of = feat.drop_duplicates("series_id").set_index("series_id")["crop_code"]
    bt = bt.assign(err=bt["actual_return"] - bt["pred_return"], crop_code=bt["series_id"].map(crop_of))
    q = bt.groupby("crop_code")["err"].quantile([0.1, 0.9]).unstack()
    last["lo_return"] = last["pred_return"] + last["crop_code"].map(q[0.1])
    last["hi_return"] = last["pred_return"] + last["crop_code"].map(q[0.9])
    return last[["series_id", "month", "price", "pred_return", "lo_return", "hi_return"]].rename(
        columns={"month": "last_month", "price": "last_price"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["bbs", "wfp"], default="bbs")
    ap.add_argument("--input", help="WFP: local CSV (default: download from HDX). BBS: clean panel CSV")
    ap.add_argument("--test-start", help="first test month, e.g. 2024-07-01 (BBS default)")
    ap.add_argument("--out", default=str(OUT), help="output folder (default: data/)")
    args = ap.parse_args()
    out = Path(args.out)

    if args.source == "bbs":
        source = args.input or BBS_CSV
        test_start = args.test_start or BBS_TEST_START
        print("Reading", source)
        m = bbs_series(source)
        meta_source = ("Bangladesh Bureau of Statistics (BBS), Monthly Statistical Bulletin, "
                       "Table 3.1 Average Wholesale Prices (Mar 2020 - Aug 2026)")
        meta_url = "https://bbs.gov.bd/"
    else:
        source = args.input or hdx_csv_url()
        test_start = args.test_start
        print("Reading", source)
        m = monthly_series(load_raw(source))
        meta_source = "World Food Programme (WFP), Bangladesh - Food Prices, Humanitarian Data Exchange"
        meta_url = f"https://data.humdata.org/dataset/{HDX_DATASET}"
    print(f"{m['series_id'].nunique()} series")

    span = (m["month"].max().to_period("M") - m["month"].min().to_period("M")).n + 1
    if span < 36:
        sys.exit(f"Only {span} months of prices: need at least 36 (the model uses 12-month changes and "
                 "needs past data to learn from). Add older bulletins.")
    feat = features(m)
    if test_start is not None and not (feat.dropna(subset=FEATURES + ["target"])["month"] >= pd.Timestamp(test_start)).any():
        sys.exit(f"No data after the test start {test_start}.")
    bt = backtest(feat, test_start)
    fc = forecast(feat, bt)

    info = m.drop_duplicates("series_id")[["series_id", "commodity", "market", "unit", "price_type"]]
    out.mkdir(parents=True, exist_ok=True)
    m[["series_id", "month", "price"]].to_csv(out / "prices.csv", index=False)
    bt.to_csv(out / "backtest.csv", index=False)
    fc.merge(info, on="series_id").to_csv(out / "forecasts.csv", index=False)
    meta = {
        "source": meta_source,
        "source_url": meta_url,
        "test_start": str(bt["month"].min().date()),
        "data_start": str(m["month"].min().date()),
        "data_end": str(m["month"].max().date()),
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "model": "Gradient Boosting on monthly price changes, pooled across series",
        "test_months": int(bt["month"].nunique()),
        "crops": sorted(m["commodity"].unique().tolist()),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Wrote {out} (prices {len(m)}, backtest {len(bt)}, forecasts {len(fc)}). Data ends {meta['data_end']}.")


if __name__ == "__main__":
    main()
