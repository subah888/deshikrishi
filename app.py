"""
DeshiKrishi: a sell-now-or-wait advisor for Bangladeshi farmers.

The app reads price history, backtest results and next-month forecasts from data/
(built by pipeline/build_data.py) and turns them into a simple decision:
sell now, or store for a month and sell later?
"""
import os
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, url_for

from core.agribot import chat as agribot_chat
from core.decision import (KG_PER_MAUND, DecisionEngine, bn_num, crop_bn,
                           market_bn, month_bn, tk)

DATA_DIR = Path(os.environ.get("DESHIKRISHI_DATA", Path(__file__).parent / "data"))
SADAI_URL = "https://sadai.gov.bd/"

STORAGE_OPTIONS = [
    (0.01, "ঘরে সংরক্ষণ (মাসে ~১%)", "Home storage (~1%/month)"),
    (0.02, "সাধারণ গুদাম (মাসে ~২%)", "Warehouse (~2%/month)"),
    (0.05, "কোল্ড স্টোরেজ / বেশি ক্ষতি (মাসে ~৫%)", "Cold storage or high losses (~5%/month)"),
]

app = Flask(__name__)
app.json.ensure_ascii = False          # Bangla stays readable in API responses
engine = DecisionEngine(DATA_DIR)

app.jinja_env.filters["bn"] = bn_num
app.jinja_env.filters["tk"] = tk
app.jinja_env.filters["crop_bn"] = crop_bn
app.jinja_env.filters["market_bn"] = market_bn
app.jinja_env.filters["month_bn"] = month_bn


@app.context_processor
def common():
    stale = None
    if engine.ready:
        end = pd.Timestamp(engine.meta["data_end"])
        months_old = (pd.Timestamp.today().to_period("M") - end.to_period("M")).n
        stale = {"end": end, "months_old": months_old} if months_old > 6 else None
    return {"ready": engine.ready, "meta": getattr(engine, "meta", {}), "stale": stale,
            "sadai_url": SADAI_URL, "KG_PER_MAUND": KG_PER_MAUND}


def _parse_quantity(form):
    try:
        qty = float(form.get("quantity", "1").translate(str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")))
    except ValueError:
        return None
    if qty <= 0 or qty > 1_000_000:
        return None
    return qty * KG_PER_MAUND if form.get("unit", "maund") == "maund" else qty


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/decide", methods=["GET", "POST"])
def decide():
    if not engine.ready:
        return render_template("no_data.html"), 503
    options = engine.series_options()
    advice, error = None, None
    form = request.form if request.method == "POST" else request.args
    selected = {
        "series": form.get("series", options[0]["id"]),
        "quantity": form.get("quantity", "20"),
        "unit": form.get("unit", "maund"),
        "storage": form.get("storage", "0.02"),
    }
    if request.method == "POST":
        qty_kg = _parse_quantity(form)
        try:
            storage = float(selected["storage"])
        except ValueError:
            storage = None
        valid_ids = {o["id"] for o in options}
        if qty_kg is None:
            error = "সঠিক পরিমাণ লিখুন (Enter a valid quantity)."
        elif storage is None or not 0 <= storage < 0.5:
            error = "সংরক্ষণ খরচ বেছে নিন (Choose a storage option)."
        elif selected["series"] not in valid_ids:
            error = "ফসল ও এলাকা বেছে নিন (Choose a crop and area)."
        else:
            advice = engine.advise(selected["series"], qty_kg, storage)
    return render_template("decide.html", options=options, storage_options=STORAGE_OPTIONS,
                           selected=selected, advice=advice, error=error)


@app.route("/prices")
def prices():
    if not engine.ready:
        return render_template("no_data.html"), 503
    options = engine.series_options()
    sid = request.args.get("series", options[0]["id"])
    if sid not in {o["id"] for o in options}:
        sid = options[0]["id"]
    hist = engine.history(sid).tail(60)          # last 5 years keeps the forecast visible
    f = engine.forecast_row(sid)
    next_month = (pd.Timestamp(f["last_month"]) + pd.offsets.MonthBegin(1))
    chart = {
        "labels": [m.strftime("%Y-%m") for m in hist["month"]] + [next_month.strftime("%Y-%m")],
        "prices": [round(float(p), 2) for p in hist["price"]] + [None],
        "forecast": [None] * (len(hist) - 1) + [round(float(f["last_price"]), 2),
                                                 round(float(f["last_price"]) * (1 + f["pred_return"]), 2)],
        "lo": [None] * len(hist) + [round(float(f["last_price"]) * (1 + f["lo_return"]), 2)],
        "hi": [None] * len(hist) + [round(float(f["last_price"]) * (1 + f["hi_return"]), 2)],
    }
    return render_template("prices.html", options=options, selected=sid, f=f, chart=chart,
                           next_month=next_month)


@app.route("/agribot")
def agribot():
    return render_template("agribot.html")


@app.route("/about")
def about():
    return render_template("about.html")


# ---------- JSON APIs ----------
@app.route("/api/chat", methods=["POST"])
def api_chat():
    body = request.get_json(silent=True) or {}
    answer, context = agribot_chat(engine, str(body.get("message", ""))[:500], str(body.get("context", ""))[:500])
    return jsonify({"response": answer, "context": context})


@app.route("/api/advice")
def api_advice():
    if not engine.ready:
        return jsonify({"error": "no data"}), 503
    sid = request.args.get("series", "")
    if sid not in {o["id"] for o in engine.series_options()}:
        return jsonify({"error": "unknown series", "series": [o["id"] for o in engine.series_options()]}), 404
    try:
        qty = float(request.args.get("quantity_kg", KG_PER_MAUND))
        storage = float(request.args.get("storage", 0.02))
    except ValueError:
        return jsonify({"error": "quantity_kg and storage must be numbers"}), 400
    if qty <= 0 or not 0 <= storage < 0.5:
        return jsonify({"error": "out of range"}), 400
    return jsonify(engine.advise(sid, qty, storage).to_dict())


@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, "data_ready": engine.ready,
                    "data_end": getattr(engine, "meta", {}).get("data_end")})


@app.errorhandler(404)
def not_found(_):
    return render_template("error.html", code=404,
                           bn="পাতাটি খুঁজে পাওয়া যায়নি।", en="Page not found."), 404


@app.errorhandler(500)
def server_error(_):
    return render_template("error.html", code=500,
                           bn="কিছু একটা ভুল হয়েছে। একটু পরে আবার চেষ্টা করুন।", en="Something went wrong."), 500


# ---------- old routes from the marketplace version ----------
@app.route("/predict")
def old_predict():
    return redirect(url_for("decide"))


@app.route("/chat")
def old_chat():
    return redirect(url_for("agribot"))


@app.route("/markets")
@app.route("/listings")
@app.route("/farmers")
@app.route("/offer")
def old_marketplace():
    return redirect(url_for("about") + "#sadai")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
