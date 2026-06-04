"""
ARGUS V3 — AI Trading Intelligence System
Terminal CLI for NSE intraday trading.

Usage:
    python argus.py                    # start interactive
    python argus.py chart.png          # analyze immediately
    python argus.py chart.png --news   # analyze with fresh news fetch
"""
import sys
import os
import time
import argparse
import textwrap
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import cv2

from config import CAPITAL, MAX_RISK_PCT, OLLAMA_MODEL
from vision.preprocessor import ImagePreprocessor
from vision.ocr           import ChartOCR
from vision.cv_analyzer   import CVAnalyzer
from intelligence.engine  import SummaryBuilder, SignalGenerator
from intelligence.brain   import IntelligenceBrain
from intelligence.llm     import OllamaClient, ReasoningEngine, autodetect_model
from news.fetcher          import NewsFetcher
from news.sentiment        import SentimentAnalyzer
from memory.store          import MemoryStore, TradeMemory


# ─────────────────────────────────────────────────────────────────────────────
#  ANSI colors
# ─────────────────────────────────────────────────────────────────────────────

def _enable_ansi():
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleMode(
                ctypes.windll.kernel32.GetStdHandle(-11), 7
            )
        except Exception:
            pass

_enable_ansi()

R="\033[0m"; B="\033[1m"; D="\033[2m"
GR="\033[32m"; RD="\033[31m"; YL="\033[33m"
CY="\033[36m"; WH="\033[37m"; MG="\033[35m"
BG_GR="\033[42m"; BG_RD="\033[41m"; BG_YL="\033[43m"; BK="\033[30m"
BG_CY="\033[46m"

def c(text, *codes): return "".join(codes)+str(text)+R

def tw():
    try: return min(os.get_terminal_size().columns, 100)
    except: return 80

TW = tw()

def hr(ch="─"): print(c(ch*TW, D))
def section(t): print(); print(c(f"  ▸ {t}", B, WH)); hr()

def row(label, val, w=22):
    print(f"{c(f'  {label:<{w}}', D)}{val}")

def badge(action):
    if action=="BUY":  return c("  BUY  ", B, BG_GR, BK)
    if action=="SELL": return c(" SELL  ", B, BG_RD, WH)
    return c(" WAIT  ", B, BG_YL, BK)

def tcolor(val):
    pos={"bullish","strong","increasing","high","above_zero","oversold","low","win"}
    neg={"bearish","weak","decreasing","very_high","overbought","below_zero","loss"}
    v=str(val).lower()
    if v in pos: return c(val, GR)
    if v in neg: return c(val, RD)
    return c(val, YL)

def stream_llm(token_gen):
    print()
    print(c("  ARGUS", B, BG_CY, BK)+"  ", end="", flush=True)
    col = 9
    for token in token_gen:
        for ch in token:
            if ch == "\n":
                print(); col = 0
            else:
                if col >= TW-4:
                    print(); print("    ", end="", flush=True); col=4
                print(ch, end="", flush=True); col+=1
    print("\n")

def wrap(text, indent=4):
    for line in text.strip().splitlines():
        if not line.strip(): print(); continue
        for seg in textwrap.wrap(line, TW-indent) or [""]:
            print(" "*indent + seg)


# ─────────────────────────────────────────────────────────────────────────────
#  Banner
# ─────────────────────────────────────────────────────────────────────────────

BANNER = r"""
    ___    ____  ______  __  _______
   /   |  / __ \/ ____/ / / / / ___/
  / /| | / /_/ / / __  / / / /\__ \
 / ___ |/ _, _/ /_/ / / /_/ /___/ /
/_/  |_/_/ |_|\____/  \____//____/   V3
"""

def print_banner(model):
    print(c(BANNER, CY, B))
    print(c(f"  NSE Intraday Intelligence  ·  Capital: ₹{CAPITAL:,.0f}  ·  Model: {model}", D))
    print(c(f"  Max risk per trade: ₹{CAPITAL*MAX_RISK_PCT:,.0f} (1%)  ·  Type 'help' for commands\n", D))


# ─────────────────────────────────────────────────────────────────────────────
#  Display functions
# ─────────────────────────────────────────────────────────────────────────────

def print_news_brief(sentiment):
    """Print morning news brief."""
    print()
    hr("═")
    print(c("  TODAY'S MARKET INTELLIGENCE", B, CY))
    hr("═")
    print(f"\n  {c('Market mood:', D)} {tcolor(sentiment.overall_signal)}  "
          f"{c(f'({sentiment.overall_score:+.2f})', D)}")
    print(f"  {c(sentiment.summary, WH)}\n")

    section("Sector Signals")
    for name, sec in sentiment.sectors.items():
        if sec.headline_count == 0:
            continue
        sig_col = GR if sec.signal=="bullish" else RD if sec.signal=="bearish" else YL
        bar = "▲" if sec.signal=="bullish" else "▼" if sec.signal=="bearish" else "─"
        stocks = ", ".join(sec.stocks[:3])
        print(f"  {c(bar, sig_col)} {c(f'{name.upper():<12}', B)} "
              f"{c(sec.signal, sig_col):<20} {c(stocks, D)}")

    if sentiment.top_bullish:
        section("Top Bullish Headlines")
        for h in sentiment.top_bullish[:3]:
            print(c(f"  ▲ ", GR) + h[:90])

    if sentiment.top_bearish:
        section("Top Bearish Headlines")
        for h in sentiment.top_bearish[:3]:
            print(c(f"  ▼ ", RD) + h[:90])
    hr()


def print_analysis(summary, signal, intel):
    """Print the full analysis panel."""
    print()
    hr("═")
    print(c("  MARKET ANALYSIS", B, CY))
    hr("═")

    conf_col = GR if summary.confidence>0.6 else YL if summary.confidence>0.35 else RD
    print(f"\n  {badge(intel.action)}  "
          f"{c(intel.conviction+' conviction', B)}  ·  "
          f"Risk: {tcolor(intel.risk)}  ·  "
          f"Confidence: {c(f'{summary.confidence:.0%}', conf_col)}")
    print(f"  {c('Trade ID:', D)} {c(intel.trade_id, MG)}\n")

    section("Intelligence Scores")
    row("Chart",   c(f"{intel.chart_score:+.2f}", GR if intel.chart_score>0 else RD))
    row("News",    c(f"{intel.news_score:+.2f}",  GR if intel.news_score>0  else RD))
    row("Memory",  c(f"{intel.memory_score:+.2f}",GR if intel.memory_score>0 else RD))
    row("Combined",c(f"{intel.combined_score:+.2f}", B,
                     GR if intel.combined_score>0 else RD if intel.combined_score<0 else YL))

    section("Chart Structure")
    row("Ticker",     c(summary.ticker or "Unknown", B))
    row("Timeframe",  summary.timeframe or "Unknown")
    row("Trend",      tcolor(f"{summary.trend} ({summary.strength})"))
    row("Pattern",    c(summary.pattern.replace("_"," "), YL) if summary.pattern else c("None", D))
    row("Momentum",   tcolor(summary.momentum))
    row("Volatility", tcolor(summary.volatility))
    row("RSI",        tcolor(f"{summary.rsi:.1f} ({summary.rsi_state})") if summary.rsi else c("N/A", D))
    row("Volume",     tcolor(summary.volume))

    section("Key Levels")
    row("Support",    c(f"₹{summary.support:.2f}", GR)  if summary.support    else c("—", D))
    row("Resistance", c(f"₹{summary.resistance:.2f}", RD) if summary.resistance else c("—", D))

    if intel.entry_price:
        section("Trade Setup  (₹4L capital, 1% risk)")
        row("Entry",    c(f"₹{intel.entry_price:.2f}", CY))
        row("Stop Loss", c(f"₹{intel.stop_price:.2f}", RD))
        row("Target",   c(f"₹{intel.target_price:.2f}", GR))
        row("Quantity", c(f"{intel.quantity} shares", B))
        row("Capital",  c(f"₹{intel.capital_required:,.0f}", WH))
        row("Max Loss", c(f"₹{intel.max_loss:,.0f}", RD))
        row("Potential",c(f"₹{intel.potential_gain:,.0f}", GR))
        row("R/R",      c(f"{intel.rr_ratio:.1f}x", B, YL))

    section("News Impact")
    for r in intel.news_reasons[:4]:
        print(c("  • ", D) + r)

    if intel.memory_reasons:
        section("Memory / History")
        for r in intel.memory_reasons[:3]:
            print(c("  • ", D) + r)

    section("Warnings")
    for w in intel.warnings:
        print(c("  ⚠  ", YL) + c(w, YL))

    section("Final Advice")
    print(c(f"  {intel.final_advice}", B, WH))
    hr()


def print_stats(memory: MemoryStore):
    stats = memory.get_stats()
    accuracy = memory.get_signal_accuracy()
    print()
    hr("═")
    print(c("  ARGUS PERFORMANCE STATS", B, CY))
    hr("═")
    print()
    row("Total trades",  c(str(stats["total"]), B))
    row("Wins",          c(str(stats["wins"]), GR))
    row("Losses",        c(str(stats["losses"]), RD))
    row("Win rate",      c(f"{stats['win_rate']:.1f}%",
                            GR if stats["win_rate"]>=55 else RD if stats["win_rate"]<45 else YL))
    row("Avg P&L/trade", c(f"₹{stats['avg_pnl']:+.2f}",
                            GR if stats["avg_pnl"]>0 else RD))
    row("Total P&L",     c(f"₹{stats['total_pnl']:+.2f}",
                            GR if stats["total_pnl"]>0 else RD))
    print()
    row("BUY accuracy",  c(f"{accuracy['BUY']['accuracy']:.0f}%", CY))
    row("SELL accuracy", c(f"{accuracy['SELL']['accuracy']:.0f}%", MG))
    hr()


def print_help():
    print()
    print(c("  ARGUS V3 Commands", B))
    hr()
    cmds = [
        ("load <path>",            "Analyze a chart screenshot"),
        ("mtf <p1> <p2> ...  ",    "Multi-timeframe: load multiple charts"),
        ("news",                   "Fetch and show today's market news"),
        ("news refresh",           "Force re-fetch news"),
        ("outcome <id> <result>",  "Record trade result: win/loss/skip"),
        ("history <TICKER>",       "Show past trades for a stock"),
        ("stats",                  "Show overall performance stats"),
        ("panel",                  "Reprint last analysis"),
        ("models",                 "List available Ollama models"),
        ("model <name>",           "Switch model"),
        ("clear",                  "Clear screen"),
        ("help",                   "Show this"),
        ("quit",                   "Exit"),
        ("",                       ""),
        ("<any question>",         "Ask ARGUS about the current chart"),
    ]
    for cmd, desc in cmds:
        if not cmd:
            print()
            continue
        print(f"  {c(cmd, CY, B):<40}{c(desc, D)}")

    print()
    print(c("  Outcome codes:", B))
    print(c("  win", GR) + "   → trade was profitable")
    print(c("  loss", RD) + "  → trade hit stop loss")
    print(c("  skip", YL) + "  → decided not to take the trade")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  Core pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_vision(img, preprocessor, ocr, cv_analyzer, sb, sg):
    print(c("\n  Running vision pipeline", D), end="", flush=True)
    t0 = time.time()
    try:
        processed  = preprocessor.preprocess(img)
        print(c(".", D), end="", flush=True)
        ocr_result = ocr.extract(processed)
        print(c(".", D), end="", flush=True)
        cv_result  = cv_analyzer.analyze(processed)
        print(c(".", D), end="", flush=True)
        summary    = sb.build(cv_result, ocr_result)
        signal     = sg.generate(summary)
        elapsed    = (time.time()-t0)*1000
        print(c(f" done ({elapsed:.0f}ms)", D))
        return summary, signal
    except Exception as e:
        print(c(f"\n  Vision error: {e}", RD))
        return None, None


def load_img(path: str):
    p = path.strip().strip('"').strip("'")
    img = cv2.imread(p)
    if img is None:
        print(c(f"  Cannot load: {p}", RD))
        return None
    print(c(f"  Loaded: {p}  ({img.shape[1]}×{img.shape[0]})", GR))
    return img


def full_analyze(img, question, preprocessor, ocr, cv_analyzer, sb, sg,
                 brain, reasoning, news_sentiment, memory):
    """Run complete pipeline and stream LLM response."""
    summary, signal = run_vision(img, preprocessor, ocr, cv_analyzer, sb, sg)
    if not summary:
        return None, None, None

    ticker = summary.ticker or "UNKNOWN"
    mem_ctx = memory.get_context_for_ticker(ticker)
    intel = brain.analyze(summary, signal, news_sentiment, memory, ticker)

    print_analysis(summary, signal, intel)
    print(c("  Generating analysis…", D))

    token_gen = reasoning.analyze(intel, summary, news_sentiment, mem_ctx, question, stream=True)
    stream_llm(token_gen)

    # Auto-save to memory
    trade = TradeMemory(
        id            = intel.trade_id,
        date          = datetime.now().strftime("%Y-%m-%d"),
        ticker        = ticker,
        timeframe     = summary.timeframe or "unknown",
        signal        = intel.action,
        conviction    = intel.conviction,
        entry_price   = intel.entry_price,
        stop_price    = intel.stop_price,
        target_price  = intel.target_price,
        chart_summary = summary.to_dict(),
        news_sentiment= news_sentiment.overall_signal if news_sentiment else "unknown",
        news_score    = intel.news_score,
        llm_reasoning = intel.final_advice,
    )
    memory.save(trade)
    print(c(f"\n  Trade saved with ID: {intel.trade_id}  (use 'outcome {intel.trade_id} win/loss' later)", D))
    return summary, signal, intel


# ─────────────────────────────────────────────────────────────────────────────
#  Main REPL
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ARGUS V3 — NSE Trading Intelligence")
    parser.add_argument("chart",   nargs="?",  help="Chart image path")
    parser.add_argument("--model", "-m",       default=None)
    parser.add_argument("--news",  action="store_true", help="Fetch fresh news on start")
    parser.add_argument("--url",   default="http://localhost:11434")
    args = parser.parse_args()

    # ── Init ──────────────────────────────────────────────────────────────────
    model = args.model or autodetect_model(args.url)
    print_banner(model)

    preprocessor = ImagePreprocessor()
    ocr          = ChartOCR()
    cv_analyzer  = CVAnalyzer()
    sb           = SummaryBuilder()
    sg           = SignalGenerator()
    brain        = IntelligenceBrain()
    llm_client   = OllamaClient(model=model, base_url=args.url)
    reasoning    = ReasoningEngine(client=llm_client)
    news_fetcher = NewsFetcher()
    sentiment_an = SentimentAnalyzer()
    memory       = MemoryStore()

    news_sentiment = None
    last_intel     = None
    img            = None

    # ── Fetch news on startup ─────────────────────────────────────────────────
    print(c("  Fetching market news…", D), end="", flush=True)
    try:
        news_items     = news_fetcher.get_news(force_refresh=args.news)
        news_sentiment = sentiment_an.analyze(news_items)
        print(c(f" {len(news_items)} headlines  [{news_sentiment.overall_signal}]", GR))
    except Exception as e:
        print(c(f" failed ({e})", YL))

    # ── Optional: analyze chart passed as arg ─────────────────────────────────
    if args.chart:
        img = load_img(args.chart)
        if img is not None:
            reasoning.reset()
            _, _, last_intel = full_analyze(
                img, "Analyze this chart. Should I enter a trade today?",
                preprocessor, ocr, cv_analyzer, sb, sg,
                brain, reasoning, news_sentiment, memory
            )

    # ── REPL ──────────────────────────────────────────────────────────────────
    while True:
        try:
            user = input(c("\n  you › ", B, CY)).strip()
        except (EOFError, KeyboardInterrupt):
            print(c("\n  Goodbye.\n", D)); break

        if not user:
            continue
        low = user.lower()

        if low in ("quit","exit","q"):
            print(c("\n  Goodbye.\n", D)); break

        elif low == "help":
            print_help()

        elif low == "clear":
            os.system("cls" if os.name=="nt" else "clear")
            print_banner(model)

        elif low == "stats":
            print_stats(memory)

        elif low == "panel":
            if last_intel and img is not None:
                summary, signal = run_vision(img, preprocessor, ocr, cv_analyzer, sb, sg)
                if summary:
                    intel = brain.analyze(summary, signal, news_sentiment, memory, summary.ticker)
                    print_analysis(summary, signal, intel)
            else:
                print(c("  No analysis yet.", YL))

        elif low in ("news", "news refresh"):
            force = "refresh" in low
            print(c("  Fetching news…", D), end="", flush=True)
            try:
                items = news_fetcher.get_news(force_refresh=force)
                news_sentiment = sentiment_an.analyze(items)
                print(c(f" {len(items)} headlines", GR))
                print_news_brief(news_sentiment)
            except Exception as e:
                print(c(f" Error: {e}", RD))

        elif low == "models":
            ms = llm_client.list_models()
            if ms:
                print(c("\n  Available models:", B))
                for m in ms:
                    mark = c("  ◀ active", GR) if model in m else ""
                    print(f"    {c(m, CY)}{mark}")
            else:
                print(c("  Cannot reach Ollama.", YL))

        elif low.startswith("model "):
            model = user[6:].strip()
            llm_client.model = model
            print(c(f"  Switched to: {model}", GR))

        elif low.startswith("load "):
            path = user[5:].strip()
            new_img = load_img(path)
            if new_img is not None:
                img = new_img
                reasoning.reset()
                _, _, last_intel = full_analyze(
                    img, "Analyze this chart. Should I enter a trade?",
                    preprocessor, ocr, cv_analyzer, sb, sg,
                    brain, reasoning, news_sentiment, memory
                )

        elif low.startswith("mtf "):
            paths = user[4:].strip().split()
            results = []
            for p in paths:
                mi = load_img(p.strip())
                if mi is not None:
                    s, sig = run_vision(mi, preprocessor, ocr, cv_analyzer, sb, sg)
                    if s:
                        results.append((p, s, sig))
            if results:
                context = "MULTI-TIMEFRAME ANALYSIS\n" + "="*40 + "\n\n"
                for p, s, sig in results:
                    intel = brain.analyze(s, sig, news_sentiment, memory, s.ticker)
                    context += f"Chart: {p}\n"
                    context += f"Signal: {intel.action} ({intel.conviction})\n"
                    context += f"Trend: {s.trend} ({s.strength})\n"
                    context += f"Score: {intel.combined_score:+.2f}\n\n"
                context += "Synthesize all timeframes. Is there confluence? What is the dominant bias and best entry?"
                reasoning._memory = [{"role": "user", "content": context}]
                from intelligence.llm import SYSTEM_PROMPT
                messages = [{"role": "system", "content": SYSTEM_PROMPT}] + reasoning._memory
                print(c("\n  Multi-timeframe synthesis…", D))
                stream_llm(llm_client.chat(messages, stream=True))

        elif low.startswith("outcome "):
            parts = user.split()
            if len(parts) >= 3:
                trade_id = parts[1]
                result   = parts[2].lower()
                entry    = float(parts[3]) if len(parts) > 3 else None
                exit_p   = float(parts[4]) if len(parts) > 4 else None
                note     = " ".join(parts[5:]) if len(parts) > 5 else None
                ok = memory.record_outcome(trade_id, result, entry, exit_p, note)
                if ok:
                    print(c(f"  Outcome recorded for {trade_id}: {result}", GR))
                    # Show updated stats
                    stats = memory.get_stats()
                    print(c(f"  Win rate: {stats['win_rate']:.1f}%  Total P&L: ₹{stats['total_pnl']:+.2f}", D))
                else:
                    print(c(f"  Trade ID {trade_id} not found.", YL))
            else:
                print(c("  Usage: outcome <trade_id> <win/loss/skip> [entry] [exit] [note]", YL))

        elif low.startswith("history "):
            ticker = user[8:].strip().upper()
            history = memory.get_ticker_history(ticker)
            trades  = memory.trades_for_ticker(ticker)
            print()
            print(c(f"  History: {ticker}", B))
            hr()
            if not trades:
                print(c(f"  No trades recorded for {ticker}", YL))
            else:
                for t in trades[:10]:
                    outcome_col = GR if t.outcome=="win" else RD if t.outcome=="loss" else YL
                    pnl_str = f"  ₹{t.pnl:+.2f}" if t.pnl else ""
                    print(f"  {c(t.date, D)}  {badge(t.signal)}  "
                          f"{c(t.conviction, WH):<12}"
                          f"{c(t.outcome or 'pending', outcome_col)}{pnl_str}")
                print()
                print(c(f"  Win rate: {history.get('win_rate',0):.1f}%  "
                        f"Total: {history.get('trades',0)} trades", D))
            hr()

        else:
            # LLM follow-up
            if img is None and last_intel is None:
                print(c("  Load a chart first: load chart.png", YL))
                continue
            if img is not None and last_intel is None:
                reasoning.reset()
                _, _, last_intel = full_analyze(
                    img, user,
                    preprocessor, ocr, cv_analyzer, sb, sg,
                    brain, reasoning, news_sentiment, memory
                )
            else:
                stream_llm(reasoning.followup(user, stream=True))


if __name__ == "__main__":
    main()
