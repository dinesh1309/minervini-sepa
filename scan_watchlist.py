"""
Quick watchlist scan — runs trend template + fundamentals on Dinesh's 10 watchlist stocks.
Outputs results to console and optionally saves to vault.

Usage:
  python scan_watchlist.py
  python scan_watchlist.py --save-vault
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '.')

from src.tools.market_data import get_stock_data
from src.tools.technical_analysis import calculate_moving_averages, check_trend_template
from src.tools.pattern_detection import detect_vcp, identify_pivot
from src.tools.market_condition import detect_market_condition, format_condition_emoji
from src.alerts.discord_alerts import send_scan_alerts
import yfinance as yf

# Dinesh's 10 watchlist stocks
WATCHLIST = {
    "india": [
        {"symbol": "SAILIFE", "name": "Sai Life Sciences", "pivot": 983, "stop": 953},
        {"symbol": "LUPIN", "name": "Lupin", "pivot": 2236, "stop": None},
        {"symbol": "ANANDRATHI", "name": "Anand Rathi Wealth", "pivot": 3230, "stop": None},
        {"symbol": "DATAPATTNS", "name": "Data Patterns", "pivot": 3215, "stop": 3280},
        {"symbol": "TDPOWERSYS", "name": "TD Power Systems", "pivot": 849, "stop": None},
    ],
    "us": [
        {"symbol": "TSM", "name": "Taiwan Semiconductor", "pivot": 385, "stop": None},
        {"symbol": "NVT", "name": "nVent Electric", "pivot": 125, "stop": None},
        {"symbol": "ADI", "name": "Analog Devices", "pivot": 360, "stop": None},
        {"symbol": "MU", "name": "Micron Technology", "pivot": None, "stop": None},
        {"symbol": "TMDX", "name": "TransMedics Group", "pivot": None, "stop": None},
    ]
}

# Index tracking
INDICES = [
    {"symbol": "^NSEI", "name": "Nifty 50", "market": "india"},
    {"symbol": "^GSPC", "name": "S&P 500", "market": "us"},
]


def get_fundamentals(symbol, market):
    """Fetch SEPA-relevant fundamental data from yfinance."""
    try:
        suffix = ".NS" if market == "india" else ""
        full = f"{symbol}{suffix}"
        ticker = yf.Ticker(full)

        info = ticker.info or {}

        # Quarterly EPS from income statement
        quarterly_eps = []
        try:
            qi = ticker.quarterly_income_stmt
            if qi is not None and not qi.empty:
                for col_name in ['Diluted EPS', 'Basic EPS']:
                    if col_name in qi.index:
                        eps_row = qi.loc[col_name].dropna().head(6)
                        for date, val in eps_row.items():
                            quarterly_eps.append({
                                "quarter": str(date)[:10],
                                "eps": round(float(val), 2)
                            })
                        break
        except Exception:
            pass

        # Calculate YoY % change for each quarter (Minervini: primary metric)
        eps_yoy_changes = []
        if len(quarterly_eps) >= 5:
            for i in range(min(4, len(quarterly_eps) - 4)):
                current_q = quarterly_eps[i]
                # Find quarter ~4 positions later (year-ago)
                yago_idx = i + 4
                if yago_idx < len(quarterly_eps):
                    year_ago_q = quarterly_eps[yago_idx]
                    yago_eps = year_ago_q["eps"]
                    curr_eps = current_q["eps"]
                    if yago_eps and yago_eps <= 0 and curr_eps > 0:
                        eps_yoy_changes.append({
                            "quarter": current_q["quarter"],
                            "current": curr_eps,
                            "year_ago": yago_eps,
                            "yoy_pct": None,
                            "note": "EASY COMP (year-ago negative)"
                        })
                    elif yago_eps and 0 < yago_eps < 1.0:
                        pct = round(((curr_eps - yago_eps) / yago_eps) * 100, 1)
                        eps_yoy_changes.append({
                            "quarter": current_q["quarter"],
                            "current": curr_eps,
                            "year_ago": yago_eps,
                            "yoy_pct": pct,
                            "note": "EASY COMP (year-ago very low)"
                        })
                    elif yago_eps and yago_eps > 0:
                        pct = round(((curr_eps - yago_eps) / yago_eps) * 100, 1)
                        eps_yoy_changes.append({
                            "quarter": current_q["quarter"],
                            "current": curr_eps,
                            "year_ago": yago_eps,
                            "yoy_pct": pct
                        })

        # EPS acceleration check (Minervini: each quarter's YoY should be >= previous)
        eps_accelerating = False
        eps_decelerating_count = 0
        if len(eps_yoy_changes) >= 2:
            for i in range(len(eps_yoy_changes) - 1):
                curr_pct = eps_yoy_changes[i].get("yoy_pct")
                prev_pct = eps_yoy_changes[i + 1].get("yoy_pct")
                if curr_pct is not None and prev_pct is not None:
                    if curr_pct < prev_pct:
                        eps_decelerating_count += 1
            if eps_yoy_changes[0].get("yoy_pct") is not None and eps_yoy_changes[-1].get("yoy_pct") is not None:
                eps_accelerating = eps_yoy_changes[0]["yoy_pct"] > eps_yoy_changes[-1]["yoy_pct"]

        # Latest quarter EPS growth
        eps_growth_yoy = eps_yoy_changes[0]["yoy_pct"] if eps_yoy_changes and eps_yoy_changes[0].get("yoy_pct") is not None else None

        # SEPA checks
        latest_eps_negative = quarterly_eps[0]["eps"] < 0 if quarterly_eps else False
        min_growth_pass = eps_growth_yoy is not None and eps_growth_yoy >= 20

        # Revenue growth
        rev_growth = info.get('revenueGrowth')
        rev_growth_pct = round(rev_growth * 100, 1) if rev_growth else None

        # Earnings growth (summary)
        earn_growth = info.get('earningsGrowth')
        earn_growth_pct = round(earn_growth * 100, 1) if earn_growth else None

        return {
            "market_cap": info.get('marketCap'),
            "pe_ratio": round(info.get('trailingPE', 0), 1) if info.get('trailingPE') else None,
            "eps_ttm": info.get('trailingEps'),
            "roe": round(info.get('returnOnEquity', 0) * 100, 1) if info.get('returnOnEquity') else None,
            "debt_equity": round(info.get('debtToEquity', 0), 1) if info.get('debtToEquity') else None,
            "profit_margin": round(info.get('profitMargins', 0) * 100, 1) if info.get('profitMargins') else None,
            "revenue_growth": rev_growth_pct,
            "earnings_growth": earn_growth_pct,
            "eps_growth_yoy_calc": eps_growth_yoy,
            "institutional_pct": round(info.get('heldPercentInstitutions', 0) * 100, 1) if info.get('heldPercentInstitutions') else None,
            "sector": info.get('sector'),
            "industry": info.get('industry'),
            "quarterly_eps": quarterly_eps[:6],
            "eps_yoy_changes": eps_yoy_changes,
            "eps_accelerating": eps_accelerating,
            "eps_decel_count": eps_decelerating_count,
            "latest_eps_negative": latest_eps_negative,
            "min_growth_pass": min_growth_pass,
            # SEPA EPS verdict
            "eps_verdict": "REJECT (negative EPS)" if latest_eps_negative
                else "REJECT (2+ decel)" if eps_decelerating_count >= 2
                else "PASS (accelerating)" if eps_accelerating and min_growth_pass
                else "PASS (>20% growth)" if min_growth_pass
                else "CAUTION (weak growth)" if eps_growth_yoy is not None and eps_growth_yoy > 0
                else "INSUFFICIENT DATA" if not eps_yoy_changes
                else "FAIL",
        }
    except Exception as e:
        return {"error": str(e)}


def scan_stock(symbol, name, market, pivot=None, stop=None, benchmark_df=None):
    """Scan a single stock for trend template."""
    exchange = "NSE" if market == "india" else None
    auto_suffix = market == "india"

    # For US stocks, use symbol directly (no suffix needed)
    if market == "us":
        full_symbol = symbol
    else:
        full_symbol = symbol

    df = get_stock_data(full_symbol, period='1y', exchange=exchange, auto_suffix=auto_suffix)

    if df is None or df.empty:
        return {"symbol": symbol, "name": name, "error": "No data", "market": market}

    # Calculate MAs
    df = calculate_moving_averages(df, [50, 150, 200])

    # Run trend template with benchmark for RS calculation
    tt = check_trend_template(df, benchmark_df=benchmark_df)

    # Current price info
    last_close = float(df['Close'].iloc[-1])
    last_date = str(df.index[-1])[:10]  # Data freshness timestamp
    sma50 = float(df['SMA_50'].iloc[-1]) if 'SMA_50' in df.columns else None
    sma150 = float(df['SMA_150'].iloc[-1]) if 'SMA_150' in df.columns else None
    sma200 = float(df['SMA_200'].iloc[-1]) if 'SMA_200' in df.columns else None
    high_52w = float(df['Close'].rolling(252).max().iloc[-1])
    low_52w = float(df['Close'].rolling(252).min().iloc[-1])

    # Distance from pivot
    dist_from_pivot = None
    if pivot:
        dist_from_pivot = round(((last_close - pivot) / pivot) * 100, 2)

    # Count passes
    checks = tt.get("checks", {})
    passed = sum(1 for v in checks.values() if str(v.get("passed", "")).lower() == "true")
    total = len(checks)

    result = {
        "symbol": symbol,
        "name": name,
        "market": market,
        "as_of": last_date,
        "price": round(last_close, 2),
        "sma50": round(sma50, 2) if sma50 else None,
        "sma150": round(sma150, 2) if sma150 else None,
        "sma200": round(sma200, 2) if sma200 else None,
        "52w_high": round(high_52w, 2),
        "52w_low": round(low_52w, 2),
        "pivot": pivot,
        "dist_from_pivot": dist_from_pivot,
        "trend_template": f"{passed}/{total}",
        "tt_pass": tt.get("all_passed", False),
        "checks": {k: str(v.get("passed", "")) for k, v in checks.items()},
        "price_vs_sma50": "ABOVE" if sma50 and last_close > sma50 else "BELOW",
        "price_vs_sma200": "ABOVE" if sma200 and last_close > sma200 else "BELOW",
        "fundamentals": get_fundamentals(symbol, market),
    }

    # VCP detection
    vcp = detect_vcp(df, symbol=symbol)
    pivot_info = identify_pivot(df, vcp)
    result["vcp"] = {
        "valid": vcp.is_valid,
        "score": vcp.score,
        "contractions": len(vcp.contractions),
        "depths": [float(d) for d in [c.depth_pct for c in vcp.contractions]],
        "volume_dry_up": vcp.volume_dry_up,
        "base_days": vcp.base_length_days,
        "detected_pivot": float(vcp.pivot_price),
        "details": vcp.details,
    }

    return result


def scan_index(symbol, name):
    """Scan an index for market condition using Minervini 4-condition model."""
    df = get_stock_data(symbol, period='1y', auto_suffix=False)

    if df is None or df.empty:
        return {"symbol": symbol, "name": name, "error": "No data"}

    mc = detect_market_condition(df, index_name=name)

    return {
        "symbol": symbol,
        "name": name,
        "price": mc.index_close,
        "as_of": mc.as_of,
        "sma50": mc.sma_50,
        "sma200": mc.sma_200,
        "price_vs_sma50": "ABOVE" if mc.above_50 else "BELOW",
        "price_vs_sma200": "ABOVE" if mc.above_200 else "BELOW",
        "condition": format_condition_emoji(mc.condition),
        "condition_raw": mc.condition,
        "confidence": mc.confidence,
        "distribution_days": mc.distribution_days_25,
        "rally_attempt_day": mc.rally_attempt_day,
        "ftd_detected": mc.ftd_detected,
        "ftd_date": mc.ftd_date,
        "details": mc.details,
    }


def format_results(index_results, stock_results, timestamp):
    """Format results as markdown for vault."""
    lines = []
    lines.append("---")
    lines.append(f"title: Watchlist Scan {timestamp[:10]}")
    lines.append(f"date: {timestamp[:10]}")
    lines.append("tags:")
    lines.append('  - "#trading"')
    lines.append('  - "#scan"')
    lines.append("type: scan-result")
    lines.append("---")
    lines.append("")
    lines.append(f"# Watchlist Scan — {timestamp[:10]}")
    lines.append("")

    # Market Conditions
    lines.append("## Market Conditions")
    lines.append("")
    lines.append("> [!warning] Market Condition Gate")
    lines.append("> No new positions in Downtrend or Rally Attempt without FTD confirmation.")
    lines.append("")
    lines.append("| Index | Price | SMA 50 | SMA 200 | Condition | Confidence | Dist Days | Rally Day | FTD? |")
    lines.append("|-------|-------|--------|---------|-----------|------------|-----------|-----------|------|")
    for idx in index_results:
        if "error" in idx:
            lines.append(f"| {idx['name']} | ERROR | — | — | — | — | — | — | — |")
        else:
            ftd = "YES" if idx.get('ftd_detected') else "NO"
            rally = idx.get('rally_attempt_day', 0) or "—"
            lines.append(f"| {idx['name']} | {idx['price']} | {idx['sma50']} | {idx['sma200']} | {idx['condition']} | {idx.get('confidence', '—')} | {idx.get('distribution_days', '—')} | {rally} | {ftd} |")

    # Detailed condition analysis
    for idx in index_results:
        if "error" not in idx and idx.get("details"):
            lines.append(f"\n**{idx['name']}:** {idx['details']}")
    lines.append("")

    # Stock Results
    for market in ["india", "us"]:
        market_stocks = [s for s in stock_results if s.get("market") == market]
        if not market_stocks:
            continue

        lines.append(f"## {'India (NSE)' if market == 'india' else 'US'} Watchlist")
        lines.append("")
        lines.append("| Stock | Price | SMA 50 | SMA 200 | vs 50 | Pivot | Dist% | TT |")
        lines.append("|-------|-------|--------|---------|-------|-------|-------|----|")

        for s in market_stocks:
            if "error" in s:
                lines.append(f"| {s['name']} | ERROR | — | — | — | — | — | — |")
            else:
                pivot_str = str(s['pivot']) if s['pivot'] else "—"
                dist_str = f"{s['dist_from_pivot']}%" if s['dist_from_pivot'] is not None else "—"
                tt_emoji = "PASS" if s['tt_pass'] else "FAIL"
                lines.append(f"| {s['name']} | {s['price']} | {s['sma50']} | {s['sma200']} | {s['price_vs_sma50']} | {pivot_str} | {dist_str} | {s['trend_template']} {tt_emoji} |")
        lines.append("")

    # Fundamentals table
    lines.append("## Fundamentals")
    lines.append("")
    lines.append("| Stock | EPS TTM | EPS Growth | Rev Growth | ROE | D/E | P/E | Margin | Inst% |")
    lines.append("|-------|---------|-----------|-----------|-----|-----|-----|--------|-------|")
    for s in stock_results:
        if "error" in s:
            lines.append(f"| {s['name']} | — | — | — | — | — | — | — | — |")
            continue
        f = s.get("fundamentals", {})
        if "error" in f:
            lines.append(f"| {s['name']} | ERROR | — | — | — | — | — | — | — |")
            continue
        eps = f.get("eps_ttm") or "—"
        eps_g = f"{f['eps_growth_yoy_calc']}%" if f.get("eps_growth_yoy_calc") is not None else (f"{f['earnings_growth']}%" if f.get("earnings_growth") is not None else "—")
        rev_g = f"{f['revenue_growth']}%" if f.get("revenue_growth") is not None else "—"
        roe = f"{f['roe']}%" if f.get("roe") is not None else "—"
        de = str(f.get("debt_equity")) if f.get("debt_equity") is not None else "—"
        pe = str(f.get("pe_ratio")) if f.get("pe_ratio") is not None else "—"
        margin = f"{f['profit_margin']}%" if f.get("profit_margin") is not None else "—"
        inst = f"{f['institutional_pct']}%" if f.get("institutional_pct") is not None else "—"
        lines.append(f"| {s['name']} | {eps} | {eps_g} | {rev_g} | {roe} | {de} | {pe} | {margin} | {inst} |")
    lines.append("")

    # Quarterly EPS YoY Analysis (Minervini style)
    lines.append("## Quarterly EPS — YoY Growth (Minervini Primary Metric)")
    lines.append("")
    lines.append("> [!info] Minervini rule: 20-25% minimum growth, look for acceleration (each quarter higher than previous). Auto-reject: 2+ consecutive quarters decelerating or negative latest EPS.")
    lines.append("")
    for s in stock_results:
        if "error" in s:
            continue
        f = s.get("fundamentals", {})
        verdict = f.get("eps_verdict", "—")
        accel = "Accelerating" if f.get("eps_accelerating") else "Decelerating" if f.get("eps_decel_count", 0) > 0 else "—"
        lines.append(f"### {s['name']} — {verdict}")
        lines.append("")

        # Always show raw EPS first
        qeps = f.get("quarterly_eps", [])
        if qeps:
            lines.append("**Raw EPS:** " + " | ".join(f"{q['quarter'][:7]}: **{q['eps']}**" for q in qeps))
            lines.append("")

        # Then show YoY analysis
        yoy = f.get("eps_yoy_changes", [])
        if yoy:
            lines.append("| Quarter | EPS | Year-Ago EPS | YoY Change | Flag |")
            lines.append("|---------|-----|-------------|-----------|------|")
            for q in yoy:
                note = q.get("note", "")
                if q.get("yoy_pct") is not None:
                    pct_str = f"{q['yoy_pct']}%"
                    flag = note if note else ("STRONG" if q['yoy_pct'] >= 40 else "OK" if q['yoy_pct'] >= 20 else "WEAK")
                else:
                    pct_str = "N/A"
                    flag = note or "—"
                lines.append(f"| {q['quarter'][:7]} | {q['current']} | {q['year_ago']} | {pct_str} | {flag} |")
            lines.append(f"\nTrend: **{accel}** | Decel quarters: {f.get('eps_decel_count', 0)}")
        elif not qeps:
            lines.append("No quarterly EPS data available")
        lines.append("")

    # VCP Pattern Detection
    lines.append("## VCP Pattern Detection")
    lines.append("")
    lines.append("| Stock | VCP? | Score | Contractions | Depths | Volume Dry? | Base Days | Detected Pivot |")
    lines.append("|-------|------|-------|-------------|--------|------------|-----------|---------------|")
    for s in stock_results:
        if "error" in s:
            lines.append(f"| {s['name']} | — | — | — | — | — | — | — |")
            continue
        v = s.get("vcp", {})
        if v.get("error"):
            lines.append(f"| {s['name']} | ERROR | — | — | — | — | — | — |")
            continue
        valid = "YES" if v.get("valid") else "NO"
        score = v.get("score", 0)
        nc = v.get("contractions", 0)
        depths = v.get("depths", [])
        depths_str = " > ".join(f"{d}%" for d in depths) if depths else "—"
        vol_dry = "YES" if v.get("volume_dry_up") else "NO"
        base = v.get("base_days", 0)
        det_pivot = v.get("detected_pivot", 0)
        lines.append(f"| {s['name']} | {valid} | {score} | {nc} | {depths_str} | {vol_dry} | {base} | {det_pivot} |")
    lines.append("")

    # VCP detail for stocks with patterns
    for s in stock_results:
        if "error" in s:
            continue
        v = s.get("vcp", {})
        if v.get("valid") or (v.get("score", 0) >= 30):
            lines.append(f"### {s['name']} — VCP Score: {v.get('score', 0)}/100")
            lines.append(f"- Contractions: {v.get('contractions', 0)}")
            depths = v.get("depths", [])
            if len(depths) >= 2:
                ratios = []
                for i in range(1, len(depths)):
                    if depths[i-1] > 0:
                        ratios.append(f"{depths[i]}/{depths[i-1]}={round(depths[i]/depths[i-1], 2)}")
                lines.append(f"- Depth sequence: {' > '.join(f'{d}%' for d in depths)}")
                if ratios:
                    lines.append(f"- Contraction ratios: {', '.join(ratios)} (ideal: ~0.5)")
            lines.append(f"- Volume dry-up: {'YES' if v.get('volume_dry_up') else 'NO'}")
            lines.append(f"- Base length: {v.get('base_days', 0)} days")
            lines.append(f"- Detected pivot: {v.get('detected_pivot', 0)}")
            if s.get("pivot"):
                lines.append(f"- Your manual pivot: {s['pivot']} (compare with detected)")
            lines.append("")

    # Detailed checks for failed stocks
    lines.append("## Trend Template Details (failed stocks)")
    lines.append("")
    for s in stock_results:
        if "error" in s or s.get("tt_pass"):
            continue
        lines.append(f"### {s['name']} ({s['symbol']}) — {s['trend_template']}")
        for check, passed in s.get("checks", {}).items():
            emoji = "PASS" if passed.lower() == "true" else "FAIL"
            lines.append(f"- {check}: {emoji}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Scan Dinesh's watchlist stocks")
    parser.add_argument("--save-vault", action="store_true", help="Save results to PVD vault")
    parser.add_argument("--alerts", action="store_true", help="Send alerts to Discord")
    parser.add_argument("--vault-path", default=None, help="Path to PVD vault (auto-detected if not set)")
    args = parser.parse_args()

    timestamp = datetime.now().isoformat()
    print(f"=== SEPA Watchlist Scan — {timestamp[:19]} ===\n")

    # Scan indices
    print("Scanning market indices...")
    index_results = []
    for idx in INDICES:
        print(f"  {idx['name']}...", end=" ", flush=True)
        result = scan_index(idx['symbol'], idx['name'])
        index_results.append(result)
        if "error" in result:
            print(f"ERROR: {result['error']}")
        else:
            print(f"{result['price']} | {result['condition']}")

    print()

    # Fetch benchmark data for RS calculation
    print("Fetching benchmarks for RS calculation...")
    nifty_df = get_stock_data('^NSEI', period='1y', auto_suffix=False)
    sp500_df = get_stock_data('^GSPC', period='1y', auto_suffix=False)
    benchmarks = {
        "india": nifty_df,
        "us": sp500_df,
    }
    print(f"  Nifty: {'OK' if nifty_df is not None else 'FAILED'}")
    print(f"  S&P 500: {'OK' if sp500_df is not None else 'FAILED'}")
    print()

    # Scan stocks
    stock_results = []
    for market, stocks in WATCHLIST.items():
        print(f"Scanning {market.upper()} watchlist...")
        benchmark = benchmarks.get(market)
        for stock in stocks:
            print(f"  {stock['name']}...", end=" ", flush=True)
            result = scan_stock(
                stock['symbol'], stock['name'], market,
                pivot=stock.get('pivot'), stop=stock.get('stop'),
                benchmark_df=benchmark
            )
            stock_results.append(result)
            if "error" in result:
                print(f"ERROR: {result['error']}")
            else:
                tt = "PASS" if result['tt_pass'] else "FAIL"
                print(f"{result['price']} | TT: {result['trend_template']} {tt}")

    # Format and display
    md = format_results(index_results, stock_results, timestamp)
    print("\n" + "=" * 60)
    print(md)

    # Save to vault
    if args.save_vault:
        vault_path = args.vault_path
        if not vault_path:
            # Try to auto-detect: check env var, then common locations
            import os
            vault_path = os.environ.get('PVD_VAULT_PATH')
            if not vault_path:
                # Check common locations
                for candidate in [
                    Path.home() / "Vibe Coding" / "PVD",
                    Path("D:/Vibe Coding/PVD"),
                    Path("/d/Vibe Coding/PVD"),
                ]:
                    if candidate.exists():
                        vault_path = candidate
                        break
            if not vault_path:
                print("WARNING: Could not find PVD vault. Use --vault-path or set PVD_VAULT_PATH env var.")
                return

        scan_dir = Path(vault_path) / "trading" / "data"
        scan_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{timestamp[:10]}-watchlist-scan.md"
        filepath = scan_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md)

        print(f"\nSaved scan to: {filepath}")

        # Update market condition log in vault
        update_market_condition_log(Path(vault_path), index_results, timestamp)

    # Send Discord alerts
    if args.alerts:
        print("\nSending Discord alerts...")
        sent = send_scan_alerts(index_results, stock_results, timestamp)
        print(f"Sent {sent} alerts to Discord")


def update_market_condition_log(vault_path, index_results, timestamp):
    """Update the vault market-condition-log.md with latest scan data."""
    log_path = vault_path / "trading" / "market-conditions" / "market-condition-log.md"

    if not log_path.exists():
        print("WARNING: market-condition-log.md not found in vault")
        return

    content = log_path.read_text(encoding="utf-8")
    date_str = timestamp[:10]

    # Find India and US index results
    india_idx = next((i for i in index_results if "Nifty" in i.get("name", "")), None)
    us_idx = next((i for i in index_results if "S&P" in i.get("name", "")), None)

    # Build the new Current Status section
    if india_idx and "error" not in india_idx:
        condition_emoji = {"UPTREND": "Confirmed Uptrend", "CAUTION": "Uptrend Under Pressure",
                          "WATCHING": "Rally Attempt", "DOWNTREND": "Downtrend"}.get(
                          india_idx.get("condition", ""), india_idx.get("condition", ""))

        rally_str = f" (Day {india_idx['rally_attempt_day']})" if india_idx.get("rally_attempt_day") else ""
        ftd_str = " | FTD DETECTED" if india_idx.get("ftd_detected") else ""

        new_status = f"""## Current Status

**As of {date_str} (auto-scan):** {condition_emoji}{rally_str}{ftd_str}
**Nifty Close:** {india_idx['price']}
**SMA 50:** {india_idx['sma50']} | **SMA 200:** {india_idx['sma200']}
**Distribution Days (25 sessions):** {india_idx.get('distribution_days', '—')}
**Details:** {india_idx.get('details', '—')}
**Action Mode:** {'Buy breakouts with progressive exposure' if 'UPTREND' in india_idx.get('condition_raw', '') else 'No new positions' if india_idx.get('condition_raw') == 'DOWNTREND' else 'Wait for FTD' if india_idx.get('condition_raw') == 'RALLY_ATTEMPT' else 'Reduce exposure, tighten stops'}

> [!warning] Market Condition Gate
> No new positions in Downtrend or Rally Attempt without FTD confirmation."""

        # Replace existing Current Status section
        import re
        pattern = r"## Current Status.*?(?=\n---)"
        content = re.sub(pattern, new_status, content, flags=re.DOTALL)

    # Append to India history table
    if india_idx and "error" not in india_idx:
        condition_icon = {"UPTREND": "UPTREND", "CAUTION": "CAUTION",
                         "WATCHING": "WATCHING", "DOWNTREND": "DOWNTREND"}.get(
                         india_idx.get("condition", ""), "?")
        rally_note = f" Day {india_idx['rally_attempt_day']}" if india_idx.get("rally_attempt_day") else ""
        new_row = f"| {date_str} | {condition_icon}{rally_note} | {india_idx['price']} | Auto-scan: {india_idx.get('distribution_days', 0)} dist days. {india_idx.get('details', '')[:80]} | — |"

        # Check if this date already exists in the history
        if date_str not in content.split("## History")[1].split("## US Market")[0] if "## History" in content else "":
            # Insert before the --- after history table
            history_end = content.find("\n---\n", content.find("## History"))
            if history_end > 0:
                content = content[:history_end] + "\n" + new_row + content[history_end:]

    # Update US Market Status section
    if us_idx and "error" not in us_idx:
        us_condition = {"UPTREND": "Confirmed Uptrend", "CAUTION": "Uptrend Under Pressure",
                       "WATCHING": "Rally Attempt", "DOWNTREND": "Downtrend"}.get(
                       us_idx.get("condition", ""), us_idx.get("condition", ""))

        us_new_row = f"| {date_str} | {us_idx['condition']} | {us_idx['price']} | Auto-scan: {us_idx.get('distribution_days', 0)} dist days. SMA50: {us_idx['sma50']}, SMA200: {us_idx['sma200']} | — |"

        # Append to US history if date not already there
        us_history_section = content.split("### US Market Condition History")[1] if "### US Market Condition History" in content else ""
        if date_str not in us_history_section:
            # Find the end of US history table
            us_note_pos = content.find("### Note on Micron")
            if us_note_pos < 0:
                us_note_pos = content.rfind("\n---\n")
            if us_note_pos > 0:
                content = content[:us_note_pos] + us_new_row + "\n\n" + content[us_note_pos:]

    # Write back
    log_path.write_text(content, encoding="utf-8")
    print(f"Updated market condition log: {log_path}")


if __name__ == "__main__":
    main()
