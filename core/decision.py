"""
Sell-now-or-wait decision engine.

A farmer holds some crop. She can sell now at today's price, or store it one month and sell
at next month's price minus a storage cost (storage, losses, interest; as a share of the price).

Rule (same as the research paper): wait only if the forecast says
    next price x (1 - storage cost) > today's price.

Honesty layer: the rule is backtested on this series' last 24 months of data. If following the
forecast would not have beaten "always sell now" at this storage cost, the app tells the farmer
to sell now and explains why.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

KG_PER_MAUND = 37.3242   # 1 maund (মণ) = 40 seer

CROP_BN = {
    "Rice (coarse)": "মোটা চাল",
    "Rice (medium grain)": "মাঝারি চাল",
    "Lentils (masur)": "মসুর ডাল",
    "Potatoes": "আলু",
    "Onions": "পেঁয়াজ",
    "Wheat flour": "আটা",
    "Oil (palm)": "পাম তেল",
}
PLACE_BN = {
    "Dhaka": "ঢাকা", "Chittagong": "চট্টগ্রাম", "Khulna": "খুলনা", "Rajshahi": "রাজশাহী",
    "Barisal": "বরিশাল", "Sylhet": "সিলেট", "Rangpur": "রংপুর", "Mymensingh": "ময়মনসিংহ",
}
MONTH_BN = ["জানুয়ারি", "ফেব্রুয়ারি", "মার্চ", "এপ্রিল", "মে", "জুন", "জুলাই", "আগস্ট",
            "সেপ্টেম্বর", "অক্টোবর", "নভেম্বর", "ডিসেম্বর"]
BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")


def bn_num(x, decimals=0):
    """Format a number with Bangla digits and thousands separators."""
    s = f"{x:,.{decimals}f}" if decimals else f"{x:,.0f}"
    return s.translate(BN_DIGITS)


def tk(x, decimals=0):
    """Taka amount in Bangla digits with a proper minus sign: −৳১,৩৫৮"""
    return ("−" if x < 0 else "") + "৳" + bn_num(abs(x), decimals)


def crop_bn(crop):
    return CROP_BN.get(crop, crop)


def market_bn(market):
    name = market.replace(" Division", "")
    bn = PLACE_BN.get(name, name)
    return f"{bn} বিভাগ" if market.endswith("Division") else bn


def month_bn(ts):
    ts = pd.Timestamp(ts)
    return f"{MONTH_BN[ts.month - 1]} {bn_num(ts.year).replace(',', '')}"


@dataclass
class Advice:
    series_id: str
    crop: str
    market: str
    decision: str                 # "wait" or "sell"
    reason: str                   # "forecast_up", "forecast_flat", "forecast_unreliable"
    quantity_kg: float
    storage_cost: float
    price_now: float              # BDT per kg
    price_next: float             # forecast, BDT per kg
    price_next_lo: float
    price_next_hi: float
    gain_tk: float                # expected extra Tk from waiting (negative = loss)
    gain_lo_tk: float
    gain_hi_tk: float
    hit_rate: float               # % of backtest months where the advice matched the best choice in hindsight
    backtest_gain_pct: float      # avg extra revenue of following advice vs always selling now
    backtest_months: int
    data_month: pd.Timestamp
    price_type: str
    extra: dict = field(default_factory=dict)

    @property
    def price_now_maund(self):
        return self.price_now * KG_PER_MAUND

    @property
    def price_next_maund(self):
        return self.price_next * KG_PER_MAUND

    def to_dict(self):
        d = {k: v for k, v in self.__dict__.items() if k != "extra"}
        d["data_month"] = str(pd.Timestamp(self.data_month).date())
        d["price_now_maund"] = self.price_now_maund
        d["price_next_maund"] = self.price_next_maund
        return d


class DecisionEngine:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.ready = all((self.data_dir / f).exists()
                         for f in ["prices.csv", "backtest.csv", "forecasts.csv", "meta.json"])
        if not self.ready:
            return
        self.prices = pd.read_csv(self.data_dir / "prices.csv", parse_dates=["month"])
        self.backtest = pd.read_csv(self.data_dir / "backtest.csv", parse_dates=["month"])
        self.forecasts = pd.read_csv(self.data_dir / "forecasts.csv", parse_dates=["last_month"])
        self.meta = json.loads((self.data_dir / "meta.json").read_text(encoding="utf-8"))
        self.forecasts = self.forecasts.sort_values(["commodity", "market"]).reset_index(drop=True)

    # ---------- lookups ----------
    def series_options(self):
        return [{"id": r.series_id, "crop": r.commodity, "market": r.market,
                 "label": f"{crop_bn(r.commodity)} — {market_bn(r.market)}",
                 "label_en": f"{r.commodity}, {r.market}"}
                for r in self.forecasts.itertuples()]

    def crops(self):
        return sorted(self.forecasts["commodity"].unique())

    def markets(self, crop=None):
        f = self.forecasts if crop is None else self.forecasts[self.forecasts["commodity"] == crop]
        return sorted(f["market"].unique())

    def find_series(self, crop, market):
        f = self.forecasts[(self.forecasts["commodity"] == crop) & (self.forecasts["market"] == market)]
        return None if f.empty else f.iloc[0]["series_id"]

    def history(self, series_id):
        h = self.prices[self.prices["series_id"] == series_id].sort_values("month")
        return h[["month", "price"]]

    def forecast_row(self, series_id):
        f = self.forecasts[self.forecasts["series_id"] == series_id]
        if f.empty:
            raise KeyError(series_id)
        return f.iloc[0]

    # ---------- backtest ----------
    def backtest_stats(self, series_id, storage_cost):
        """How the sell/wait rule would have done in this series' test months."""
        b = self.backtest[self.backtest["series_id"] == series_id]
        if b.empty:
            return float("nan"), float("nan"), 0
        wait_rule = (1 + b["pred_return"]) * (1 - storage_cost) > 1
        wait_value = (1 + b["actual_return"]) * (1 - storage_cost)        # per Tk of today's price
        best_is_wait = wait_value > 1
        revenue = np.where(wait_rule, wait_value, 1.0)
        hit_rate = float((wait_rule == best_is_wait).mean() * 100)
        gain_pct = float((revenue.mean() - 1) * 100)
        return hit_rate, gain_pct, len(b)

    # ---------- the decision ----------
    def advise(self, series_id, quantity_kg, storage_cost=0.02):
        f = self.forecast_row(series_id)
        p = float(f["last_price"])
        r, lo, hi = float(f["pred_return"]), float(f["lo_return"]), float(f["hi_return"])
        c = float(storage_cost)

        def gain(ret):
            return quantity_kg * p * ((1 + ret) * (1 - c) - 1)

        hit_rate, bt_gain, n = self.backtest_stats(series_id, c)
        forecast_says_wait = (1 + r) * (1 - c) > 1
        reliable = not (bt_gain <= 0)          # NaN counts as reliable (no evidence either way)

        if forecast_says_wait and reliable:
            decision, reason = "wait", "forecast_up"
        elif forecast_says_wait:
            decision, reason = "sell", "forecast_unreliable"
        else:
            decision, reason = "sell", "forecast_flat"

        return Advice(
            series_id=series_id, crop=f["commodity"], market=f["market"],
            decision=decision, reason=reason, quantity_kg=quantity_kg, storage_cost=c,
            price_now=p, price_next=p * (1 + r), price_next_lo=p * (1 + lo), price_next_hi=p * (1 + hi),
            gain_tk=gain(r), gain_lo_tk=gain(lo), gain_hi_tk=gain(hi),
            hit_rate=hit_rate, backtest_gain_pct=bt_gain, backtest_months=n,
            data_month=f["last_month"], price_type=f["price_type"],
        )
