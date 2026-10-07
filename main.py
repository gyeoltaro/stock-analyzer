"""사용법:
  python main.py report   [--date YYYYMMDD] [--universe 200] [--top 20]
  python main.py backtest [--years 5] [--universe 100] [--top 20]
"""
import argparse
import os
from pathlib import Path

import pandas as pd

from analyzer import backtest, data
from analyzer.report import backtest_report, daily_report
from analyzer.scoring import ScoreConfig, score

REPORTS = Path(__file__).parent / "reports"


def cmd_report(args):
    date = data.latest_business_day(args.date)
    print(f"기준일 {date}, 유니버스 조회 중...")
    uni = data.universe(date, args.universe)
    close = data.close_prices(uni.index, data.lookback_start(date, 420), date)
    fund = data.fundamentals(date)
    cfg = ScoreConfig(top_n=args.top)
    ranked = score(close, fund, cfg)
    md = daily_report(date, ranked, uni["name"], len(uni), args.top)
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"{pd.Timestamp(date):%Y-%m-%d}.md"
    out.write_text(md, encoding="utf-8")
    (REPORTS / "latest.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"저장: {out}")


def cmd_backtest(args):
    end = data.latest_business_day(args.date)
    test_start = data.lookback_start(end, int(args.years * 365))
    fetch_start = data.lookback_start(test_start, 420)  # 첫 리밸런싱에 필요한 과거 데이터
    print(f"백테스트 {test_start}~{end}, 유니버스 조회 중...")
    uni = data.universe(end, args.universe)
    close = data.close_prices(uni.index, fetch_start, end)
    bench = data.benchmark(fetch_start, end)

    cache = {}

    def fund_at(day):
        key = data.ymd(day)
        if key not in cache:
            try:
                cache[key] = data.fundamentals(key)
            except Exception as e:
                print(f"[warn] {key} 펀더멘털 실패: {e}")
                cache[key] = None
        return cache[key]

    res = backtest.run(close, fund_at, ScoreConfig(top_n=args.top), bench, cost=args.cost, start=test_start)
    md = backtest_report(res.summary(), res.holdings, uni["name"], f"{test_start}~{end}")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "backtest.md").write_text(md, encoding="utf-8")
    print(md)


def main():
    p = argparse.ArgumentParser(description="한국 주식 규칙 기반 분석")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="오늘의 종목 점수 리포트")
    r.add_argument("--date")
    r.add_argument("--universe", type=int, default=200)
    r.add_argument("--top", type=int, default=20)
    r.set_defaults(func=cmd_report)
    b = sub.add_parser("backtest", help="월간 리밸런싱 백테스트")
    b.add_argument("--date")
    b.add_argument("--years", type=float, default=5)
    b.add_argument("--universe", type=int, default=100)
    b.add_argument("--top", type=int, default=20)
    b.add_argument("--cost", type=float, default=0.003)
    b.set_defaults(func=cmd_backtest)
    args = p.parse_args()
    if not (os.getenv("KRX_ID") and os.getenv("KRX_PW")):
        print("[주의] KRX_ID / KRX_PW 환경 변수가 없습니다. data.krx.co.kr 계정이 필요합니다.")
    args.func(args)


if __name__ == "__main__":
    main()
