"""
AgriBot: a rule-based Bangla/English assistant with one job, answering
"should I sell now or wait?" using the same decision engine as the website.

It understands crop names, places, and quantities like "২০ মণ" or "500 kg".
It does not invent prices: every number comes from the data files.
"""
import re

from .decision import KG_PER_MAUND, bn_num, crop_bn, market_bn, month_bn, tk

CROP_WORDS = {
    "Rice (coarse)": ["চাল", "ধান", "rice", "paddy"],
    "Lentils (masur)": ["মসুর", "ডাল", "lentil", "masur"],
    "Potatoes": ["আলু", "potato"],
    "Onions": ["পেঁয়াজ", "পিঁয়াজ", "onion"],
}
PLACE_WORDS = {
    "Dhaka": ["ঢাকা", "dhaka"], "Chittagong": ["চট্টগ্রাম", "chittagong", "chattogram"],
    "Khulna": ["খুলনা", "khulna"], "Rajshahi": ["রাজশাহী", "rajshahi"],
    "Barisal": ["বরিশাল", "barisal", "barishal"], "Sylhet": ["সিলেট", "sylhet"],
    "Rangpur": ["রংপুর", "rangpur"], "Mymensingh": ["ময়মনসিংহ", "mymensingh"],
}
BN_TO_EN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
GREETINGS = ["hello", "hi", "salam", "সালাম", "হ্যালো", "আসসালামু"]

HELP = ("আমি বলতে পারি এখন বিক্রি করবেন নাকি এক মাস অপেক্ষা করবেন। "
        "ফসল, জায়গা আর পরিমাণ লিখুন। যেমন: \"রংপুরে ২০ মণ চাল এখন বেচব?\"")


def _find(text, table):
    for key, words in table.items():
        if any(w in text for w in words):
            return key
    return None


def _quantity_kg(text):
    t = text.translate(BN_TO_EN_DIGITS).replace(",", "")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(মণ|মন|maund|mound|কেজি|kg|কেজ)", t)
    if not m:
        return None
    n, unit = float(m.group(1)), m.group(2)
    return n * KG_PER_MAUND if unit in ("মণ", "মন", "maund", "mound") else n


def reply(engine, message, storage_cost=0.02):
    if not engine.ready:
        return "দুঃখিত, এখনো দামের ডেটা লোড হয়নি। (No price data yet: run the data pipeline.)"
    text = message.lower().strip()
    if not text:
        return HELP

    crop = _find(text, CROP_WORDS)
    place = _find(text, PLACE_WORDS)

    if crop is None:
        if any(g in text for g in GREETINGS):
            return "আসসালামু আলাইকুম! " + HELP
        crops = ", ".join(crop_bn(c) for c in engine.crops())
        return f"কোন ফসলের কথা বলছেন? এখন যেগুলোর তথ্য আছে: {crops}।"

    if crop not in engine.crops():
        return f"{crop_bn(crop)}-এর যথেষ্ট দামের তথ্য এখনো নেই। আছে: " + ", ".join(crop_bn(c) for c in engine.crops()) + "।"

    markets = engine.markets(crop)
    if place is None:
        return (f"কোন এলাকায় বিক্রি করবেন? {crop_bn(crop)}-এর তথ্য আছে: "
                + ", ".join(market_bn(m) for m in markets) + "।")

    market = next((m for m in markets if m == f"{place} Division"), None) or \
        next((m for m in markets if m == place), None)
    if market is None:
        return (f"{market_bn(place)}-এর {crop_bn(crop)}-এর তথ্য নেই। আছে: "
                + ", ".join(market_bn(m) for m in markets) + "।")

    qty = _quantity_kg(text) or KG_PER_MAUND
    a = engine.advise(engine.find_series(crop, market), qty, storage_cost)

    paddy_note = ""
    if crop == "Rice (coarse)" and any(w in text for w in ("ধান", "paddy")):
        paddy_note = ("নোট: ধানের আলাদা দামের তথ্য নেই, তাই চালের দাম দিয়ে হিসাব করা হলো। "
                      "দাম বাড়বে না কমবে সেটা কাজে লাগবে, কিন্তু টাকার অঙ্ক ধানের জন্য কম হবে।\n")
    head = paddy_note + f"{market_bn(market)}, {crop_bn(crop)} ({month_bn(a.data_month)}-এর দাম অনুযায়ী):\n"
    price = (f"এখনকার দাম মণপ্রতি {tk(a.price_now_maund)}। "
             f"পরের মাসের পূর্বাভাস {tk(a.price_next_maund)}।\n")
    if a.decision == "wait":
        verdict = (f"👉 এক মাস অপেক্ষা করা লাভজনক হতে পারে: {bn_num(qty / KG_PER_MAUND, 1)} মণে "
                   f"আনুমানিক {tk(a.gain_tk)} বেশি (খরচ বাদে)।")
    elif a.reason == "forecast_unreliable":
        verdict = ("👉 এখনই বিক্রি করুন। পূর্বাভাস দাম বাড়ার কথা বলছে, কিন্তু অতীতের পরীক্ষায় "
                   "এই এলাকার এই ফসলে পূর্বাভাস মেনে চলে লাভ হয়নি।")
    else:
        verdict = "👉 এখনই বিক্রি করা ভালো: সংরক্ষণ খরচ বাদ দিলে অপেক্ষা করে লাভ হবে না বলে পূর্বাভাস।"
    trust = (f"\nঅতীতের {bn_num(a.backtest_months)} মাসে এই পরামর্শ "
             f"{bn_num(a.hit_rate)}% ক্ষেত্রে সঠিক ছিল।") if a.backtest_months else ""
    return head + price + verdict + trust
