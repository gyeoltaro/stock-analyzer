"""사용법:
  python main.py report   [--date YYYYMMDD] [--universe 0] [--top 20]
  python main.py backtest [--years 5] [--universe 0] [--top 20]
  python main.py swing           # 오늘의 단기 스윙 신호
  python main.py swing-backtest [--years 3]

--universe 0 은 전 종목, N 이면 시가총액 상위 N개.
"""
import argparse
from pathlib import Path

import pandas as pd

from analyzer import backtest, data, swing
from analyzer.report import backtest_report, daily_report, swing_backtest_report, swing_report
from analyzer.scoring import ScoreConfig, score, technical_candidates

REPORTS = Path(__file__).parent / "reports"


def cmd_report(args):
    date = data.latest_business_day(args.date)
    print(f"기준일 {date}, 유니버스 조회 중...")
    uni = data.universe(date, args.universe)
    close = data.close_prices(uni.index, data.lookback_start(date, 420), date)
    cfg = ScoreConfig(top_n=args.top)
    candidates = technical_candidates(close, cfg)
    print(f"가격 조건 통과 {len(candidates)}개, 재무지표 조회 중...")
    fund = data.fundamentals(date, candidates)
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
    if not data.has_point_in_time_fundamentals():
        print("[안내] 네이버 데이터는 과거 PER/PBR이 없어 백테스트에서 가치 팩터를 제외합니다.")

    def fund_at(day):
        if not data.has_point_in_time_fundamentals():
            return None
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


def cmd_swing(args):
    date = data.latest_business_day(args.date)
    print(f"기준일 {date}, 유니버스 조회 중...")
    uni = data.universe(date, args.universe)
    px = data.ohlcv_panel(uni.index, data.lookback_start(date, 260), date)
    sig = swing.today_signals(px, swing.SwingConfig(), args.top)
    md = swing_report(date, sig, uni["name"], len(uni))
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"swing-{pd.Timestamp(date):%Y-%m-%d}.md").write_text(md, encoding="utf-8")
    (REPORTS / "swing-latest.md").write_text(md, encoding="utf-8")
    print(md)


def cmd_swing_backtest(args):
    end = data.latest_business_day(args.date)
    test_start = data.lookback_start(end, int(args.years * 365))
    fetch_start = data.lookback_start(test_start, 220)  # 120일선 계산용
    print(f"스윙 백테스트 {test_start}~{end}, 유니버스 조회 중...")
    uni = data.universe(end, args.universe)
    px = data.ohlcv_panel(uni.index, fetch_start, end)
    bench = data.benchmark(fetch_start, end)
    cfg = swing.SwingConfig(max_positions=args.positions, cost_roundtrip=args.cost)
    results = {s: swing.backtest(px, s, cfg, start=test_start, benchmark=bench) for s in swing.SETUPS}
    md = swing_backtest_report(results, f"{test_start}~{end}", cfg)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "swing-backtest.md").write_text(md, encoding="utf-8")
    print(md)


def main():
    p = argparse.ArgumentParser(description="한국 주식 규칙 기반 분석")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="오늘의 종목 점수 리포트")
    r.add_argument("--date")
    r.add_argument("--universe", type=int, default=0, help="0=전 종목")
    r.add_argument("--top", type=int, default=20)
    r.set_defaults(func=cmd_report)
    b = sub.add_parser("backtest", help="월간 리밸런싱 백테스트")
    b.add_argument("--date")
    b.add_argument("--years", type=float, default=5)
    b.add_argument("--universe", type=int, default=0, help="0=전 종목")
    b.add_argument("--top", type=int, default=20)
    b.add_argument("--cost", type=float, default=0.003)
    b.set_defaults(func=cmd_backtest)
    sw = sub.add_parser("swing", help="오늘의 단기 스윙 신호")
    sw.add_argument("--date")
    sw.add_argument("--universe", type=int, default=0, help="0=전 종목")
    sw.add_argument("--top", type=int, default=10)
    sw.set_defaults(func=cmd_swing)
    sb = sub.add_parser("swing-backtest", help="단기 스윙 백테스트")
    sb.add_argument("--date")
    sb.add_argument("--years", type=float, default=3)
    sb.add_argument("--universe", type=int, default=0, help="0=전 종목")
    sb.add_argument("--positions", type=int, default=5)
    sb.add_argument("--cost", type=float, default=0.005)
    sb.set_defaults(func=cmd_swing_backtest)
    args = p.parse_args()
    print(f"데이터 소스: {data.source_name()}")
    args.func(args)


if __name__ == "__main__":
    main()
