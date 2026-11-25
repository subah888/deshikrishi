"""
Extract Table 3.1 (Average Wholesale Prices of Selected Agricultural Commodities)
from BBS Monthly Statistical Bulletin PDFs into one long CSV.

Usage: python extract_bbs.py pdfs/*.pdf --out bbs_wholesale_raw.csv
Needs: pdftotext (poppler-utils)
"""
import argparse
import re
import subprocess
from pathlib import Path

import pandas as pd

TOWNS = ["Dhaka", "Narayanganj", "Chattogram", "Chittagong", "Rajshahi", "Khulna",
         "Sylhet", "Rangpur", "Barishal", "Barisal", "Mymensingh"]
TOWN_NORM = {"Chittagong": "Chattogram", "Barisal": "Barishal"}
COMMODITIES = [  # (regex, name) checked in order
    (r"\bRice\b", "Rice"), (r"\bMasur\b", "Masur"), (r"\bKhesari\b", "Khesari"),
    (r"\bMoong\b", "Moong"), (r"\bGram\b", "Gram"), (r"\bMustard\b", "Mustard"),
    (r"\bChil+ies\b", "Chilies (dry)"), (r"\bOnion\b", "Onion"), (r"\bGarlic\b", "Garlic"),
    (r"\bTurmeric\b", "Turmeric"), (r"\bGinger\b", "Ginger"), (r"\bCoriander\b", "Coriander seed"),
    (r"\bPotato\b", "Potato"), (r"\bBrinjal\b", "Brinjal"), (r"\bEgg\b", "Egg"),
    (r"\bTobacco\b", "Tobacco leaf"), (r"\bBetel\s*nut|\bBetelnut\b", "Betelnut"),
    (r"\bBetel\s*leaf\b", "Betel leaf"), (r"\bGroundnut\b", "Groundnut"),
]
RICE_GRADE = [(r"Fine", "Fine"), (r"Medium", "Medium"), (r"Coarse", "Coarse")]
MONTH_RE = re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?,?\s*'?(\d{2})\b")
NUM_RE = re.compile(r"^-?\d[\d,]*(\.\d+)?$|^-$")


def nums(text):
    return [t.replace(",", "") for t in text.split() if NUM_RE.match(t.replace(",", ""))]


def has_town(line):
    return any(re.search(rf"\b{t}\b", line) for t in TOWNS)


def page_texts(pdf):
    n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout).group(1))
    for i in range(1, n + 1):
        t = subprocess.run(["pdftotext", "-f", str(i), "-l", str(i), "-layout", pdf, "-"],
                           capture_output=True, text=True).stdout
        yield i, t


def parse_pdf(pdf):
    rows, months, commodity, grade = [], None, None, None
    for page, text in page_texts(pdf):
        if not re.search(r"3\.1\s+Average Wholesale Prices of Selected Agri", text, re.I):
            continue
        if "CONTENTS" in text.upper()[:400]:
            continue
        lines = text.splitlines()
        for idx, line in enumerate(lines):
            found = MONTH_RE.findall(line)
            if len(found) >= 6:
                months = [pd.Timestamp(f"20{y}-{m}-01") for m, y in found[-6:]]
                continue
            for pat, name in COMMODITIES:
                if re.search(pat, line, re.I):
                    if name != commodity:
                        commodity, grade = name, None
                    break
            if commodity == "Rice":
                for pat, g in RICE_GRADE:
                    if re.search(rf"\b{pat}\b", line):
                        grade = g
            town = next((t for t in TOWNS if re.search(rf"\b{t}\b", line)), None)
            if not town or months is None or commodity is None:
                continue
            tokens = nums(line.split(town, 1)[1])
            if len(tokens) < 6:   # row wrapped onto the line below or above
                nxt = lines[idx + 1] if idx + 1 < len(lines) else ""
                prv = lines[idx - 1] if idx > 0 else ""
                if not has_town(nxt) and len(tokens) + len(nums(nxt)) >= 6:
                    tokens = tokens + nums(nxt)
                elif not has_town(prv) and not MONTH_RE.search(prv) and len(tokens) + len(nums(prv)) >= 6:
                    tokens = tokens + nums(prv)
            if len(tokens) < 6:
                continue
            for m, v in zip(months, tokens[-6:]):
                rows.append({
                    "issue": Path(pdf).stem, "page": page,
                    "commodity": commodity if commodity != "Rice" else f"Rice ({grade or 'NA'})",
                    "town": TOWN_NORM.get(town, town), "month": m,
                    "price_tk_per_quintal": None if v == "-" else float(v),
                })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="+")
    ap.add_argument("--out", default="bbs_wholesale_raw.csv")
    a = ap.parse_args()
    all_rows = []
    for p in sorted(a.pdfs):
        r = parse_pdf(p)
        print(f"{Path(p).name}: {len(r)} values")
        all_rows += r
    df = pd.DataFrame(all_rows)
    df.to_csv(a.out, index=False)
    print("Wrote", a.out, len(df))


if __name__ == "__main__":
    main()
