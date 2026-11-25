"""Build the clean monthly panel from bbs_wholesale_raw.csv.
Suspect rule: within one issue, if the last two months exactly copy months 3-4 AND months 3-4 are not
simply flat prices (i.e. the row varies), the last two values are flagged as suspected copy errors and dropped.
Short gaps (<=2 months) are interpolated in the modelling step, not here."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
ap = argparse.ArgumentParser()
ap.add_argument("--raw", default=str(RAW / "bbs_wholesale_raw.csv"))
ap.add_argument("--outdir", default=str(RAW))
args = ap.parse_args()
OUTDIR = Path(args.outdir)
d = pd.read_csv(args.raw, parse_dates=["month"])
CROPS = ["Rice (Coarse)", "Rice (Medium)", "Masur", "Onion", "Potato"]
TOWNS = ["Dhaka", "Chattogram", "Rajshahi", "Khulna", "Sylhet", "Rangpur"]
d = d[d.commodity.isin(CROPS) & d.town.isin(TOWNS)].copy()
d["suspect_copy"] = False
for (i, c, t), g in d.groupby(["issue", "commodity", "town"]):
    g = g.sort_values("month"); v = g.price_tk_per_quintal.values
    if len(v) == 6 and v[4] == v[2] and v[5] == v[3] and len(set(v[2:4]) | set(v[:2])) > 1:
        d.loc[g.index[4:], "suspect_copy"] = True
clean = d[~d.suspect_copy & d.price_tk_per_quintal.notna()]
# each month appears in exactly one issue; keep one value per series-month
panel = clean.groupby(["commodity", "town", "month"], as_index=False)["price_tk_per_quintal"].first()
full = pd.MultiIndex.from_product([CROPS, TOWNS, pd.date_range(d.month.min(), d.month.max(), freq="MS")],
                                  names=["commodity", "town", "month"])
panel = panel.set_index(["commodity", "town", "month"]).reindex(full).reset_index()
# typo rule: a value more than 2.5x above or below the other towns' median in the same month is a digit slip
med = panel.groupby(["commodity", "month"]).price_tk_per_quintal.transform("median")
ratio = panel.price_tk_per_quintal / med
panel["suspect_typo"] = (ratio > 2.5) | (ratio < 0.4)
print("typos flagged:"); print(panel[panel.suspect_typo][["commodity", "town", "month", "price_tk_per_quintal"]].to_string(index=False))
panel.loc[panel.suspect_typo, "price_tk_per_quintal"] = np.nan
panel["price_tk_per_kg"] = panel.price_tk_per_quintal / 100
panel["price_tk_per_maund"] = panel.price_tk_per_kg * 37.3242
panel.to_csv(OUTDIR / "bbs_wholesale_monthly_clean.csv", index=False)
d.to_csv(OUTDIR / "bbs_wholesale_flagged.csv", index=False)
print("flagged as copy:", int(d.suspect_copy.sum()), "of", len(d))
print("panel:", panel.shape, "| missing:", panel.price_tk_per_quintal.isna().sum(),
      "| months:", panel.month.min().date(), "to", panel.month.max().date())
print(panel.groupby("commodity").price_tk_per_quintal.apply(lambda s: s.isna().sum()))
