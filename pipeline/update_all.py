"""
One command to refresh DeshiKrishi when a new BBS Monthly Statistical Bulletin comes out.

1. Put the bulletin PDFs (old + new) in one folder, e.g. bulletins/
   (download from bbs.gov.bd -> প্রকাশনা -> পরিসংখ্যান বুলেটিন; the February and August issues are enough,
   each covers six months)
2. Run:  python pipeline/update_all.py bulletins/
3. Check the printed summary, run the tests (pytest), commit data/ and push.

Needs pdftotext/pdfinfo (poppler-utils): Ubuntu `sudo apt install poppler-utils`, macOS `brew install poppler`,
Windows: install poppler and add its bin folder to PATH.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def run(*cmd):
    print("\n$", " ".join(str(c) for c in cmd))
    subprocess.run([sys.executable, *map(str, cmd)], check=True, cwd=ROOT)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    pdfs = sorted(Path(sys.argv[1]).glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs in {sys.argv[1]}")
    run(ROOT / "pipeline" / "extract_bbs.py", *pdfs, "--out", RAW / "bbs_wholesale_raw.csv")
    run(ROOT / "pipeline" / "clean_bbs.py")
    run(ROOT / "pipeline" / "build_data.py")
    print("\nDone. Next: pytest, then commit data/ and push.")


if __name__ == "__main__":
    main()
