"""네이버 금융 데이터 주소 점검용 (개발 디버깅)."""
import json
import re
import sys

sys.path.insert(0, ".")
from analyzer.data_naver import _session  # noqa: E402

URLS = [
    "https://m.stock.naver.com/api/stocks/marketValue/KOSPI?page=1&pageSize=3",
    "https://m.stock.naver.com/api/stocks/marketValue/KOSDAQ?page=1&pageSize=2",
    "https://m.stock.naver.com/api/stock/005930/integration",
    "https://m.stock.naver.com/api/stock/005930/basic",
    "https://api.stock.naver.com/stock/005930/basic",
]
for u in URLS:
    print("=" * 20, u)
    try:
        r = _session.get(u, timeout=15)
        print("status", r.status_code, r.headers.get("content-type"))
        try:
            print(json.dumps(r.json(), ensure_ascii=False)[:2500])
        except ValueError:
            print(r.text[:500])
    except Exception as e:
        print("error", e)

print("=" * 20, "market_sum html")
r = _session.get("https://finance.naver.com/sise/sise_market_sum.naver?sosok=0&page=1", timeout=15)
html = r.text
for key in ("__NEXT_DATA__", "itemCode", "005930", "/api/"):
    idx = [m.start() for m in re.finditer(re.escape(key), html)][:3]
    print(key, len(idx))
    for i in idx[:2]:
        print("   ...", html[max(0, i - 200): i + 400].replace("\n", " "))
