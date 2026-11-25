"""Tests for the decision engine, AgriBot and Flask routes, on small hand-made data."""
import importlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

UP = "Rice_(coarse)|Rangpur_Division|KG"        # forecast up, forecast reliable
UNRELIABLE = "Lentils_(masur)|Dhaka_Division|KG"  # forecast up, but lost money in backtest
FLAT = "Rice_(coarse)|Dhaka_Division|KG"         # forecast flat


@pytest.fixture()
def data_dir(tmp_path):
    months = pd.date_range("2018-01-01", periods=30, freq="MS")
    rows = []
    for sid in (UP, UNRELIABLE, FLAT):
        rows += [{"series_id": sid, "month": m, "price": 40 + i * 0.1} for i, m in enumerate(months)]
    pd.DataFrame(rows).to_csv(tmp_path / "prices.csv", index=False)

    bt = []
    for i, m in enumerate(months[-4:]):
        bt.append({"series_id": UP, "month": m, "price": 40, "pred_return": 0.05, "actual_return": 0.06})     # always right
        bt.append({"series_id": UNRELIABLE, "month": m, "price": 40, "pred_return": 0.05, "actual_return": -0.05})  # always wrong
        bt.append({"series_id": FLAT, "month": m, "price": 40, "pred_return": 0.0, "actual_return": 0.0})
    pd.DataFrame(bt).to_csv(tmp_path / "backtest.csv", index=False)

    fc = [
        {"series_id": UP, "last_month": "2020-06-01", "last_price": 40.0, "pred_return": 0.05, "lo_return": 0.0,
         "hi_return": 0.1, "commodity": "Rice (coarse)", "market": "Rangpur Division", "unit": "KG", "price_type": "Retail"},
        {"series_id": UNRELIABLE, "last_month": "2020-06-01", "last_price": 100.0, "pred_return": 0.05, "lo_return": -0.02,
         "hi_return": 0.1, "commodity": "Lentils (masur)", "market": "Dhaka Division", "unit": "KG", "price_type": "Retail"},
        {"series_id": FLAT, "last_month": "2020-06-01", "last_price": 40.0, "pred_return": 0.0, "lo_return": -0.03,
         "hi_return": 0.03, "commodity": "Rice (coarse)", "market": "Dhaka Division", "unit": "KG", "price_type": "Retail"},
    ]
    pd.DataFrame(fc).to_csv(tmp_path / "forecasts.csv", index=False)
    (tmp_path / "meta.json").write_text(json.dumps({
        "source": "test", "source_url": "https://example.org", "data_start": "2018-01-01",
        "data_end": "2020-06-01", "built_at": "2026-11-19", "test_months": 4}))
    return tmp_path


@pytest.fixture()
def engine(data_dir):
    from core.decision import DecisionEngine
    return DecisionEngine(data_dir)


@pytest.fixture()
def client(data_dir, monkeypatch):
    monkeypatch.setenv("DESHIKRISHI_DATA", str(data_dir))
    import app as app_module
    importlib.reload(app_module)
    app_module.app.testing = True
    return app_module.app.test_client()


# ---------- decision engine ----------
def test_wait_when_forecast_up_and_reliable(engine):
    a = engine.advise(UP, quantity_kg=100, storage_cost=0.02)
    assert a.decision == "wait" and a.reason == "forecast_up"
    assert a.gain_tk == pytest.approx(100 * 40 * (1.05 * 0.98 - 1))
    assert a.hit_rate == 100 and a.backtest_months == 4


def test_sell_when_forecast_unreliable(engine):
    a = engine.advise(UNRELIABLE, quantity_kg=100, storage_cost=0.02)
    assert a.decision == "sell" and a.reason == "forecast_unreliable"
    assert a.backtest_gain_pct < 0 and a.hit_rate == 0


def test_sell_when_forecast_flat(engine):
    a = engine.advise(FLAT, quantity_kg=100, storage_cost=0.02)
    assert a.decision == "sell" and a.reason == "forecast_flat"
    assert a.gain_tk < 0


def test_high_storage_cost_flips_to_sell(engine):
    assert engine.advise(UP, 100, storage_cost=0.01).decision == "wait"
    assert engine.advise(UP, 100, storage_cost=0.06).decision == "sell"


def test_gain_range_is_ordered(engine):
    a = engine.advise(UP, 37.3242, 0.02)
    assert a.gain_lo_tk <= a.gain_tk <= a.gain_hi_tk
    assert a.price_now_maund == pytest.approx(40 * 37.3242)


def test_engine_without_data(tmp_path):
    from core.decision import DecisionEngine
    assert DecisionEngine(tmp_path).ready is False


# ---------- AgriBot ----------
def test_agribot_full_question(engine):
    from core.agribot import reply
    r = reply(engine, "রংপুরে ২০ মণ চাল এখন বেচব?")
    assert "রংপুর" in r and "অপেক্ষা" in r


def test_agribot_asks_for_place(engine):
    from core.agribot import reply
    assert "কোন এলাকায়" in reply(engine, "চাল বেচব?")


def test_agribot_english_and_unreliable(engine):
    from core.agribot import reply
    assert "এখনই বিক্রি" in reply(engine, "should I sell lentil in dhaka 500 kg")


def test_agribot_unknown_crop_and_paddy_note(engine):
    from core.agribot import reply
    assert "তথ্য" in reply(engine, "রংপুরে আলু বেচব?")
    assert "ধানের আলাদা দামের তথ্য নেই" in reply(engine, "রংপুরে ১০ মণ ধান")


def test_agribot_quantity_parsing():
    from core.agribot import _quantity_kg
    assert _quantity_kg("২০ মণ") == pytest.approx(20 * 37.3242)
    assert _quantity_kg("500 kg") == 500
    assert _quantity_kg("কিছু চাল") is None


# ---------- routes ----------
def test_pages_render(client):
    for path in ["/", "/decide", "/prices", "/agribot", "/about", f"/prices?series={UP}"]:
        r = client.get(path)
        assert r.status_code == 200, path
    assert "গবেষণা ডেমো" in client.get("/").get_data(as_text=True)   # stale-data banner


def test_decide_post(client):
    r = client.post("/decide", data={"series": UP, "quantity": "২০", "unit": "maund", "storage": "0.02"})
    html = r.get_data(as_text=True)
    assert r.status_code == 200 and "এক মাস অপেক্ষা করুন" in html


def test_decide_post_rejects_bad_input(client):
    r = client.post("/decide", data={"series": UP, "quantity": "abc", "unit": "kg", "storage": "0.02"})
    assert "সঠিক পরিমাণ" in r.get_data(as_text=True)
    r = client.post("/decide", data={"series": "nope", "quantity": "5", "unit": "kg", "storage": "0.02"})
    assert "ফসল ও এলাকা" in r.get_data(as_text=True)


def test_api_advice_and_chat(client):
    r = client.get(f"/api/advice?series={UP}&quantity_kg=100&storage=0.02")
    assert r.status_code == 200 and r.get_json()["decision"] == "wait"
    assert client.get("/api/advice?series=nope").status_code == 404
    assert client.get(f"/api/advice?series={UP}&storage=x").status_code == 400
    r = client.post("/api/chat", json={"message": "ঢাকায় চাল"})
    assert "এখনই বিক্রি" in r.get_json()["response"]


def test_old_routes_redirect(client):
    assert client.get("/predict").status_code == 302
    assert client.get("/markets").headers["Location"].endswith("/about#sadai")


def test_no_data_page(tmp_path, monkeypatch):
    monkeypatch.setenv("DESHIKRISHI_DATA", str(tmp_path))
    import app as app_module
    importlib.reload(app_module)
    c = app_module.app.test_client()
    assert c.get("/decide").status_code == 503
    assert c.get("/").status_code == 200


# ---------- BBS data (real files in data/) ----------
REAL = ROOT / "data"


@pytest.mark.skipif(not (REAL / "forecasts.csv").exists(), reason="data not built")
def test_real_bbs_data_and_agribot():
    from core.agribot import reply
    from core.decision import DecisionEngine
    e = DecisionEngine(REAL)
    assert {"Rice (Coarse)", "Potato", "Onion", "Masur"} <= set(e.crops())
    for o in e.series_options():
        a = e.advise(o["id"], 100, 0.02)
        assert a.decision in ("wait", "sell") and a.backtest_months > 0
    assert "রংপুর" in reply(e, "রংপুরে ২০ মণ চাল বেচব?")
    assert "চট্টগ্রাম" in reply(e, "chattogram onion 500 kg")
    assert "মাঝারি চাল" in reply(e, "ঢাকায় মাঝারি চাল")


# ---------- follow-up memory, health, errors ----------
def test_agribot_remembers_crop_then_place(engine):
    from core.agribot import chat
    a1, ctx = chat(engine, "চাল বেচব?")
    assert "কোন এলাকায়" in a1 and ctx
    a2, ctx2 = chat(engine, "রংপুর", ctx)
    assert "রংপুর" in a2 and "অপেক্ষা" in a2 and ctx2 == ""


def test_agribot_remembers_place_then_crop(engine):
    from core.agribot import chat
    a1, ctx = chat(engine, "ঢাকা বিভাগে বেচব")
    assert "কোন ফসলের" in a1 and ctx
    a2, _ = chat(engine, "মসুর ৫ মণ", ctx)
    assert "মসুর" in a2 and "এখনই বিক্রি" in a2


def test_new_question_overrides_context(engine):
    from core.agribot import chat
    _, ctx = chat(engine, "চাল বেচব?")
    a, _ = chat(engine, "ঢাকা বিভাগে মসুর", ctx)
    assert "মসুর" in a


def test_chat_api_returns_context(client):
    r = client.post("/api/chat", json={"message": "চাল বেচব?"}).get_json()
    assert r["context"]
    r2 = client.post("/api/chat", json={"message": "রংপুর", "context": r["context"]}).get_json()
    assert "রংপুর" in r2["response"] and r2["context"] == ""


def test_health_and_404(client):
    h = client.get("/healthz").get_json()
    assert h["ok"] and h["data_ready"]
    r = client.get("/does-not-exist")
    assert r.status_code == 404 and "খুঁজে পাওয়া যায়নি" in r.get_data(as_text=True)
