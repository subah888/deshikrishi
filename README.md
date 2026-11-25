# DeshiKrishi · দেশীকৃষি

**Should a farmer sell the harvest now, or store it for a month?**

DeshiKrishi combines a machine-learning price forecast with the farmer's storage cost and answers that question in Bangla, with the expected gain or loss in taka and how often the advice was right in past data.

It is the prototype behind a research project on the *decision value* of crop price forecasts in Bangladesh: not just "how accurate is the forecast?" but "does following it earn farmers more money?"

Live demo: https://deshikrishi.onrender.com

## Features

| Page | What it does |
|---|---|
| ⚖️ **বেচব না রাখব?** (`/decide`) | Crop, area, quantity (মণ or kg) and storage type → sell now or wait, expected gain or loss, 80% range, backtested hit rate |
| 📈 **দামের ধারা** (`/prices`) | Five-year price chart with next month's forecast and range |
| 🤖 **AgriBot** (`/agribot`) | Bangla or English questions, e.g. *"রংপুরে ২০ মণ চাল এখন বেচব?"*; asks back for a missing crop or area |
| ℹ️ **পদ্ধতি** (`/about`) | Method, data source, limitations |

Market directories and today's official prices are left to the Department of Agricultural Marketing's [Sadai](https://sadai.gov.bd/) platform. DeshiKrishi focuses only on the sell-or-wait decision.

## How the decision works

1. **Forecast:** a Gradient Boosting model predicts next month's price change from recent changes, last year's seasonal change, distance from the 6-month average and the time of year, trained on all crop–area series together.
2. **Rule:** wait only if `forecast price × (1 − storage cost) > today's price`.
3. **Honesty check:** the rule is backtested from July 2024 onward with rolling-origin evaluation (retrained every month on past data only). Where following the forecast would have earned less than selling right away, the app recommends selling now and explains why.

## Data

- **Source:** Bangladesh Bureau of Statistics (BBS), *Monthly Statistical Bulletin*, Table 3.1 "Average Wholesale Prices of Selected Agricultural Commodities", 13 issues (August 2020 – August 2026). Monthly prices from March 2020 to August 2026 for coarse rice, medium rice, masur dal, onion and potato in Dhaka, Chattogram, Rajshahi, Khulna, Sylhet and Rangpur.
- **Extraction and cleaning:** values were extracted from the bulletin PDFs by script, each with its issue and page (`data/raw/bbs_wholesale_raw.csv`). 108 values copied from earlier months and 10 digit typos were removed; every removed value is listed in `data/raw/bbs_wholesale_flagged.csv`.
- **Limitations:** wholesale prices are used as a signal for farm-gate price movements; farmers receive somewhat less. The data is for rice (চাল), while farmers sell paddy (ধান).

## Research

`research/DeshiKrishi_Step3_BBS_Analysis.ipynb` compares Naive, Seasonal, Random Forest and Gradient Boosting forecasts on accuracy and on decision value (sell now vs store one month), before (July 2023 – June 2024) and during the volatile period after July 2024. Model outputs are in `research/results/`; the paper's figures and tables in `research/figures/` are generated from those outputs by `research/make_figures.py`.

## Project structure

```
app.py                    Flask web app
core/decision.py          sell-or-wait engine and backtest statistics
core/agribot.py           Bangla/English assistant using the same engine
pipeline/extract_bbs.py   extracts Table 3.1 from BBS bulletin PDFs
pipeline/clean_bbs.py     removes copied months and digit typos
pipeline/build_data.py    model training, backtest and forecasts
research/                 analysis notebook, results, figures
templates/                Bangla-first pages
tests/                    automated tests
```

## Author

Subah Sadia Tabassum, B.Sc. Digital Business and Innovation, Tokyo International University.
