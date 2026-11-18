# DeshiKrishi · দেশীকৃষি

**Should a farmer sell the harvest now, or store it for a month?**

DeshiKrishi combines a machine-learning price forecast with the farmer's storage cost and answers
that question in Bangla, with the expected gain or loss in taka and how often the advice was right in past data.

It is the prototype behind a research project on the *decision value* of crop price forecasts in Bangladesh:
not just "how accurate is the forecast?" but "does following it earn farmers more money?"

## Features

| Page | What it does |
|---|---|
| ⚖️ **বেচব না রাখব?** (`/decide`) | Pick crop, area, quantity (মণ or kg) and storage type → sell now or wait, expected gain/loss, 80% range, backtested hit rate |
| 📈 **দামের ধারা** (`/prices`) | Five-year price chart with next month's forecast and range |
| 🤖 **AgriBot** (`/agribot`) | Ask in Bangla or English: *"রংপুরে ২০ মণ চাল এখন বেচব?"* |
| ℹ️ **পদ্ধতি** (`/about`) | Method, data source, limitations |
| API | `GET /api/advice?series=...&quantity_kg=...&storage=0.02`, `POST /api/chat` |

For market directories and today's official prices, the app points users to the Department of Agricultural
Marketing's [Sadai](https://sadai.gov.bd/) platform. DeshiKrishi focuses only on the sell-or-wait decision.

## How the decision works

1. **Forecast:** a Gradient Boosting model predicts next month's price change from recent changes, last year's
   seasonal change, distance from the 6-month average and the time of year. One model is trained on all crop–area
   series together.
2. **Rule:** wait only if `forecast price × (1 − storage cost) > today's price`.
3. **Honesty check:** the rule is backtested on each series' last 24 months (rolling origin: retrained every month
   on past data only). If following the forecast would have earned *less* than always selling right away,
   the app recommends selling now and says why.

## Run it

```bash
# 1. Build the data (downloads WFP Bangladesh food prices, trains the model, ~1 minute)
pip install -r requirements-pipeline.txt
python pipeline/build_data.py
# or, with a CSV you downloaded from https://data.humdata.org/dataset/wfp-food-prices-for-bangladesh
python pipeline/build_data.py --input wfp_food_prices_bgd.csv

# 2. Start the app
pip install -r requirements.txt
python app.py            # http://localhost:5000
```

Commit the files in `data/` so a deployed copy (the `Procfile` runs `gunicorn app:app`) has the data without
retraining. Re-run step 1 whenever new prices are available.

Tests: `pip install -r requirements-dev.txt && pytest`

## Project layout

```
app.py                 Flask routes
core/decision.py       sell-or-wait engine, backtest stats, Bangla formatting
core/agribot.py        rule-based Bangla/English assistant using the same engine
pipeline/build_data.py data download, model training, backtest, forecasts → data/
templates/             Bangla-first pages (Bootstrap, Chart.js)
tests/                 pytest suite
```

## Data and limitations

- **Source:** World Food Programme, *Bangladesh – Food Prices*, Humanitarian Data Exchange.
- **Old data:** WFP's division-level series end in 2020–2021, so this is a research demo, not live market advice.
  The app shows a warning banner when data is more than 6 months old.
- **Retail prices** are used as a signal for farm-gate price movements; farmers receive less than retail.
- **Paddy vs rice:** farmers sell paddy (ধান); the data is for rice (চাল). AgriBot says so when asked about paddy.
- Planned: farmer-received (farm-gate) prices from the Department of Agricultural Marketing for recent years.

## Author

Subah Sadia Tabassum, B.Sc. Digital Business and Innovation, Tokyo International University.
