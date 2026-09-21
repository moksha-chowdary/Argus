"""
ARGUS V4 — Jarvis Terminal
Complete NSE intraday intelligence system.

Fixes in this version:
  - LLM memory resets between different stock analyses (no cross-stock bleed)
  - Real OHLCV indicators replace pixel-brightness guessing
  - yfinance error spam completely suppressed
  - Session clock (9:15 morning → power hour → closing warning)
  - Folder watcher for auto-analysis
  - Price alerts
  - EMA / RSI / MACD displayed in signal panel

Usage:
  python argus.py                      start clean
  python argus.py chart.png            analyze immediately
  python argus.py --no-live            live feed OFF
  python argus.py --train              training mode
  python argus.py --watch C:\charts    auto-watch folder
"""

import sys, os, json, argparse, time, logging, warnings
from datetime import datetime

# Suppress noisy libraries BEFORE any imports
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("chromadb").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (BANNER, CAPITAL, MAX_RISK_PCT, STATE_PATH,
                    LIVE_FEED_DEFAULT, OLLAMA_MODEL, WATCH_FOLDER)
from vision.analyzer import ChartAnalyzer
from intelligence.brain import ArgusBrain
from intelligence.llm import OllamaClient
from intelligence.price_tracker import PriceTracker
from intelligence.indicators import calculate as calc_indicators
from intelligence.watcher import FolderWatcher
from intelligence.alerts import AlertManager, _beep
from news.fetcher import NewsFetcher
from news.scheduler import NewsScheduler
from evaluation.backtest import WalkForwardBacktest, print_backtest_report
from memory.store import ArgusMemory, TradeRecord, new_trade_id

# ── ANSI ──────────────────────────────────────────────────────────────────────
R  = "\033[91m"; G = "\033[92m"; Y = "\033[93m"
B  = "\033[94m"; C = "\033[96m"; M = "\033[95m"
W  = "\033[97m"; DM= "\033[2m";  RS= "\033[0m"; BLD="\033[1m"

def col(text, colour=W): return f"{colour}{text}{RS}"
def hr(ch="─", n=58):    return col(ch*n, DM)


# ─────────────────────────────────────────────────────────────────────────────
#  State
# ─────────────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    try:
        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
        with open(STATE_PATH) as f:
            return json.load(f)
    except Exception:
        return {"live_feed": LIVE_FEED_DEFAULT, "watch_folder": WATCH_FOLDER}

def save_state(state: dict):
    try:
        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
        with open(STATE_PATH, "w") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
#  Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_banner(live_feed: bool):
    os.system("cls" if sys.platform == "win32" else "clear")
    print(col(BANNER, C))
    status = col("● LIVE", G) if live_feed else col("○ OFFLINE", Y)
    print(f"  Status: {status}  |  Capital: {col('₹'+format(CAPITAL,','), W)}  |  Risk/trade: {col('1%', Y)}")
    print(hr())

def print_signal(sig, stock_name: str = ""):
    colour = G if sig.action=="BUY" else (R if sig.action=="SELL" else Y)
    session_col = Y if sig.session in ("morning","closing") else G

    print(f"\n{hr()}")
    print(f"  {col('ARGUS SIGNAL', BLD)}" + (f"  — {col(stock_name, W)}" if stock_name else ""))
    print(hr())

    print(f"  Action     : {col(sig.action, colour+BLD)}  |  Grade: {col(sig.grade,W)}  |  "
          f"Confidence: {col(str(sig.confidence)+'%', W)}  |  "
          f"Session: {col(sig.session, session_col)}")
    print(f"  Conviction : {sig.conviction}  |  R/R: {col(sig.rr_ratio or 'N/A', W)}")
    
    # V5 ML Probabilities & Drift Telemetry
    p_up_c = G if sig.prob_online >= 0.55 else (R if sig.prob_online <= 0.45 else Y)
    dl_c   = G if sig.prob_dl >= 0.55 else (R if sig.prob_dl <= 0.45 else Y)
    drift_c = R if sig.drift_status != "stable" else G
    print(f"  ML Prob    : River Online={col(f'{sig.prob_online:.1%}', p_up_c)}  |  "
          f"PyTorch LSTM={col(f'{sig.prob_dl:.1%}', dl_c)}  |  "
          f"ADWIN={col(sig.drift_status.upper(), drift_c)}")
    if sig.prediction_id:
        print(f"  Audit      : ID={col(sig.prediction_id, C)}  as-of={col(sig.features_asof, DM)}")
    print()

    # Live price
    if sig.live_price:
        chg_c = G if sig.live_change_pct >= 0 else R
        sign  = "+" if sig.live_change_pct >= 0 else ""
        print(f"  Live price : {col('₹'+str(sig.live_price), W)}  "
              f"({col(sign+str(sig.live_change_pct)+'%', chg_c)})")

    # Indicators row
    ind_parts = []
    if sig.ema_20:  ind_parts.append(f"EMA20={col(str(sig.ema_20),W)}")
    if sig.ema_50:  ind_parts.append(f"EMA50={col(str(sig.ema_50),W)}")
    if sig.rsi:
        rc = G if 40<sig.rsi<60 else (R if sig.rsi>=70 or sig.rsi<=30 else Y)
        ind_parts.append(f"RSI={col(str(sig.rsi), rc)}")
    if sig.macd_cross != "none":
        mc = G if "bullish" in sig.macd_cross else R
        ind_parts.append(col(f"MACD:{sig.macd_cross}", mc))
    if ind_parts:
        print(f"  Indicators : {' | '.join(ind_parts)}")

    # Patterns
    if sig.patterns:
        print(f"  Patterns   : {col(', '.join(sig.patterns), DM)}")

    print()
    print(f"  Entry      : {col('₹'+str(sig.entry_price) if sig.entry_price else 'TBD', W)}")
    print(f"  Stop loss  : {col('₹'+str(sig.stop_price)  if sig.stop_price  else 'TBD', R)}")
    print(f"  Target     : {col('₹'+str(sig.target_price) if sig.target_price else 'TBD', G)}")
    print()
    print(f"  Quantity   : {col(str(sig.quantity)+' shares', W)}")
    print(f"  Capital    : {col('₹'+format(int(sig.capital_required),',') if sig.capital_required else 'N/A', W)}")
    print(f"  Max loss   : {col('₹'+format(int(sig.max_loss),',') if sig.max_loss else 'N/A', R)}")
    print()
    print(f"  Scores     : chart={col(f'{sig.chart_score:+.2f}',W)}  "
          f"indic={col(f'{sig.indicator_score:+.2f}',W)}  "
          f"news={col(f'{sig.news_score:+.2f}',W)}  "
          f"memory={col(f'{sig.memory_score:+.2f}',W)}  "
          f"combined={col(f'{sig.combined_score:+.2f}',W)}")

    if sig.warnings:
        print()
        for w_ in sig.warnings:
            print(f"  {col('⚠ '+w_, Y)}")

    print(f"\n  {hr('─',50)}")
    print(f"  {col(sig.final_advice, DM)}")
    print(hr())

def print_live_snapshot(tracker: PriceTracker):
    snap = tracker.snapshot()
    if not snap:
        print(col("  No live data yet — tracker may still be fetching.", DM))
        return
    print(f"\n{hr()}")
    print(f"  {'TICKER':<18} {'PRICE':>10} {'CHG%':>8} {'HIGH':>10} {'LOW':>10}")
    print(hr("─"))
    for key in ("^NSEI", "^NSEBANK"):
        data = snap.pop(key, None)
        if data:
            name  = "NIFTY 50" if key=="^NSEI" else "BANK NIFTY"
            chg_c = G if data["change_pct"]>=0 else R
            sign  = "+" if data["change_pct"]>=0 else ""
            print(f"  {col(name,BLD):<26} {col('₹'+str(data['price']),W):>18} "
                  f"{col(sign+str(data['change_pct'])+'%',chg_c):>14} "
                  f"{str(data['high']):>10} {str(data['low']):>10}")
    print(hr("─"))
    for ticker, data in sorted(snap.items()):
        chg_c = G if data["change_pct"]>=0 else R
        sign  = "+" if data["change_pct"]>=0 else ""
        name  = ticker.replace(".NS","")
        print(f"  {name:<18} {col('₹'+str(data['price']),W):>18} "
              f"{col(sign+str(data['change_pct'])+'%',chg_c):>14} "
              f"{str(data['high']):>10} {str(data['low']):>10}")
    upd = tracker.last_update()
    if upd:
        print(col(f"\n  Last updated: {upd}", DM))
    print(hr())

def print_alerts(alert_mgr: AlertManager):
    alerts = alert_mgr.list_alerts()
    if not alerts:
        print(col("  No active alerts.", DM))
        return
    print(f"\n{hr()}")
    print(f"  {col('ACTIVE ALERTS', BLD)}")
    print(hr("─"))
    for a in alerts:
        print(f"  {col(a.id, C)}  {a.ticker.replace('.NS','')}  "
              f"{col(a.direction.upper(), W)}  {col('₹'+str(a.price), Y)}"
              f"{'  '+a.note if a.note else ''}")
    print(hr())

def print_stats(mem: ArgusMemory):
    s = mem.stats()
    print(f"\n{hr()}")
    print(f"  {col('ARGUS PERFORMANCE DASHBOARD', BLD)}")
    print(hr("─"))
    print(f"  Total analyses : {s['total_trades']}")
    print(f"  Resolved       : {s['resolved']}")
    wr_c = G if s['win_rate']>=0.6 else (Y if s['win_rate']>=0.4 else R)
    print(f"  Win rate       : {col(str(int(s['win_rate']*100))+'%', wr_c)}")
    pnl_c = G if s['total_pnl']>=0 else R
    print(f"  Total P&L      : {col('₹'+format(int(s['total_pnl']),','), pnl_c)}")
    if s['by_stock']:
        print(f"\n  Per-stock breakdown:")
        for st, d in s['by_stock'].items():
            total = d['wins']+d['losses']
            wr    = d['wins']/total if total else 0
            wc    = G if wr>=0.6 else (Y if wr>=0.4 else R)
            print(f"    {st:<22} {d['wins']}W / {d['losses']}L  "
                  f"{col(str(int(wr*100))+'%', wc)}")
    print(hr())

def print_news(sentiment):
    print(f"\n{hr()}")
    print(f"  {col('MARKET SENTIMENT', BLD)}  ({col(sentiment.fetched_at, DM)})")
    print(hr("─"))
    sc = G if sentiment.overall_signal=="bullish" else (R if sentiment.overall_signal=="bearish" else Y)
    print(f"  Overall: {col(sentiment.overall_signal.upper(), sc+BLD)}")
    print()
    for sec, data in sentiment.sectors.items():
        dc = G if data.signal=="bullish" else (R if data.signal=="bearish" else Y)
        bar = "+" if data.signal=="bullish" else ("-" if data.signal=="bearish" else "~")
        print(f"  {bar} {sec:<12} {col(data.signal, dc)}")
    print()
    for h_ in sentiment.top_headlines[:4]:
        print(col(f"  • {h_[:85]}", DM))
    print(hr())

def print_help():
    cmds = [
        ("load <path>",                    "Analyze a chart image"),
        ("prices",                         "Live NSE watchlist"),
        ("news",                           "Today's sector sentiment"),
        ("livefeed on/off",                "Toggle live prices + news"),
        ("watch on [folder]",              "Auto-analyze charts dropped in folder"),
        ("watch off",                      "Stop folder watcher"),
        ("alert <STOCK> above/below <₹>", "Set price alert (e.g. alert ICICI above 1320)"),
        ("alerts",                         "List active alerts"),
        ("alert clear",                    "Remove all alerts"),
        ("alert remove <id>",              "Remove specific alert"),
        ("outcome <id> win/loss [price]",  "Record trade outcome"),
        ("pending",                        "Trades awaiting outcome"),
        ("stats",                          "Performance dashboard"),
        ("history <STOCK>",                "Past trades for a stock"),
        ("train",                          "Training mode"),
        ("reload",                         "Reload rulebook.json"),
        ("backtest [STOCK]",               "Run walk-forward validation (zero leakage)"),
        ("drift",                          "Show recent ADWIN concept drift events"),
        ("retrain",                        "Retrain PyTorch LSTM on feature store"),
        ("news_scan",                      "Trigger FinBERT RSS scrape & sentiment cycle"),
        ("model <name>",                   "Switch LLM model"),
        ("reset",                          "Clear chat memory"),
        ("clear",                          "Clear screen"),
        ("help",                           "This menu"),
        ("quit",                           "Exit"),
    ]
    print(f"\n{hr()}")
    for cmd, desc in cmds:
        print(f"  {col(cmd, C):<48} {col(desc, DM)}")
    print(hr())


# ─────────────────────────────────────────────────────────────────────────────
#  Core analysis function
#  FIX: LLM memory resets per-analysis so no cross-stock bleed
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(path, analyzer, brain, llm, mem, tracker,
                 news_fetcher, alert_mgr, live_feed, quiet=False) -> str:
    """
    Analyze a chart image.
    Returns the trade ID.
    Key fix: llm.reset() is called BEFORE building the prompt so each stock
    gets a completely fresh LLM context. Follow-up questions after analysis
    will still have context for THAT stock specifically.
    """
    if not os.path.exists(path):
        print(col(f"  File not found: {path}", R))
        return ""

    stock_name = os.path.splitext(os.path.basename(path))[0]
    print(col(f"\n  Analysing {os.path.basename(path)}...", DM), end="", flush=True)

    # ── Vision layer ──────────────────────────────────────────────────────────
    summary = analyzer.analyze(path)
    if not summary:
        print(col(" could not read image.", R))
        return ""

    # Use filename as ticker fallback if OCR didn't find one
    if not summary.ticker:
        fname = os.path.splitext(os.path.basename(path))[0].upper()
        # Strip common suffixes like _daily, _15m, _1h
        for suffix in ["_DAILY","_15M","_1H","_4H","_WEEKLY","_1D"]:
            fname = fname.replace(suffix,"")
        summary.ticker = fname[:12] if fname else "UNKNOWN"

    ticker     = summary.ticker
    ticker_ns  = ticker + ".NS" if not ticker.endswith(".NS") else ticker

    # ── Indicators (real OHLCV) ───────────────────────────────────────────────
    indicators = None
    if live_feed:
        print(col(" fetching candles...", DM), end="", flush=True)
        df = tracker.fetch_candles(ticker_ns, interval="15m", period="5d")
        if df is not None:
            indicators = calc_indicators(df)
            print(col(" done", G))
        else:
            print(col(" no candle data (chart-only mode)", Y))
    else:
        print(col(" offline mode", Y))

    # ── Supporting data ───────────────────────────────────────────────────────
    sentiment  = news_fetcher.fetch() if live_feed else None
    live_price = tracker.get(ticker_ns) if live_feed else None
    stock_acc  = mem.stock_accuracy(ticker)

    # ── Brain ─────────────────────────────────────────────────────────────────
    sig = brain.generate_signal(
        chart_summary=summary,
        news_sentiment=sentiment,
        memory_context=stock_acc,
        live_price=live_price,
        indicators=indicators,
        live_feed_on=live_feed,
        ticker=ticker,
        candle_df=df if live_feed else None,
    )

    print_signal(sig, ticker)

    if stock_acc.get("samples", 0) > 0:
        wr_c = G if stock_acc["win_rate"] >= 0.5 else R
        print(f"  Memory: {stock_acc['samples']} past trades on {ticker} — "
              f"win rate {col(str(int(stock_acc['win_rate']*100))+'%', wr_c)}\n")

    if sentiment and live_feed:
        print_news(sentiment)

    # ── Save trade ────────────────────────────────────────────────────────────
    tid = new_trade_id()
    rec = TradeRecord(
        id=tid, stock=ticker,
        date=datetime.now().strftime("%Y-%m-%d"),
        action=sig.action, entry=sig.entry_price,
        stop=sig.stop_price, target=sig.target_price,
        confidence=sig.confidence,
        patterns=sig.patterns,
        news_signal=sentiment.overall_signal if sentiment else "N/A",
        outcome="pending",
    )
    mem.save_trade(rec)
    print(col(f"  Trade ID: {tid}  →  'outcome {tid} win/loss <exit_price>' when resolved\n", DM))

    # ── KEY FIX: Reset LLM memory, inject fresh stock context ────────────────
    llm.reset()

    # Build tightly-scoped prompt — only THIS stock's data
    ind_context = ""
    if indicators:
        ind_context = (
            f"EMA20={indicators.ema_20} EMA50={indicators.ema_50} "
            f"RSI={indicators.rsi}({indicators.rsi_state}) "
            f"MACD_cross={indicators.macd_cross} "
            f"Volume={indicators.volume_signal}({indicators.volume_ratio:.1f}x) "
            f"Higher_lows={indicators.higher_lows} Lower_highs={indicators.lower_highs} "
            f"Session={indicators.session}"
        )

    prompt = (
        f"Analyzing {ticker} exclusively. This is the ONLY stock being discussed.\n"
        f"Trend: {summary.trend} ({summary.strength}) | "
        f"Pattern: {summary.pattern} | Volume: {summary.volume}\n"
        f"{ind_context}\n"
        f"Signal: {sig.action} ({sig.conviction}) | Confidence: {sig.confidence}%\n"
        f"Entry: ₹{sig.entry_price} | Stop: ₹{sig.stop_price} | Target: ₹{sig.target_price}\n"
        f"Patterns: {', '.join(sig.patterns) if sig.patterns else 'none'}\n"
        f"Warnings: {', '.join(sig.warnings) if sig.warnings else 'none'}\n"
        f"News: {sentiment.overall_signal if sentiment else 'unavailable'}\n"
        f"Give a 4-5 sentence intraday assessment for {ticker} only. "
        f"State entry rationale, key risk, and what would invalidate this trade."
    )

    if not quiet:
        print(col(f"  ARGUS on {ticker}:\n", BLD))
        for tok in llm.chat(prompt):
            print(tok, end="", flush=True)
        print("\n")

    return tid


# ─────────────────────────────────────────────────────────────────────────────
#  Training mode
# ─────────────────────────────────────────────────────────────────────────────

def training_mode(analyzer, brain, llm, mem, tracker, live_feed):
    print(col("\n  ═══ TRAINING MODE ═══", M+BLD))
    print(col("  Upload chart → ARGUS predicts → reveal outcome → ARGUS learns", DM))
    print(col("  Type 'done' to exit\n", DM))

    while True:
        path = input(col("  chart path (or 'done'): ", M)).strip().strip('"').strip("'")
        if path.lower() == "done":
            break
        if not os.path.exists(path):
            print(col(f"  Not found: {path}", R))
            continue

        summary = analyzer.analyze(path)
        if not summary:
            print(col("  Could not read image.", R))
            continue

        if not summary.ticker:
            summary.ticker = os.path.splitext(os.path.basename(path))[0].upper()[:10]

        # Get indicators if live
        indicators = None
        if live_feed:
            df = tracker.fetch_candles(summary.ticker + ".NS", interval="15m", period="5d")
            if df is not None:
                indicators = calc_indicators(df)

        sig = brain.generate_signal(
            chart_summary=summary, news_sentiment=None,
            memory_context={}, indicators=indicators,
            live_feed_on=False, ticker=summary.ticker or "",
        )
        print_signal(sig, summary.ticker or "")

        # Fresh LLM context for this training sample
        llm.reset()
        prompt = (
            f"TRAINING — {summary.ticker}. "
            f"Trend: {summary.trend}. Pattern: {summary.pattern}. "
            f"Signal: {sig.action} {sig.confidence}%. "
            f"EMA stack: {indicators.ema_stack if indicators else 'unknown'}. "
            f"RSI: {indicators.rsi if indicators else 'unknown'}. "
            f"In 3 bullets: prediction reasoning + key risk."
        )
        print(col("\n  ARGUS prediction:\n", BLD))
        for tok in llm.chat(prompt):
            print(tok, end="", flush=True)
        print("\n")

        outcome_raw = input(col("  Outcome (win/loss/skip): ", M)).strip().lower()
        if outcome_raw in ("win","loss","skip"):
            tid = new_trade_id()
            rec = TradeRecord(
                id=tid, stock=summary.ticker or "UNK",
                date=datetime.now().strftime("%Y-%m-%d"),
                action=sig.action, entry=sig.entry_price,
                stop=sig.stop_price, target=sig.target_price,
                confidence=sig.confidence, patterns=sig.patterns,
                news_signal="N/A", outcome=outcome_raw,
            )
            mem.save_trade(rec)
            mem.record_outcome(tid, outcome_raw)

            # Self-adjustment — still in same llm context for this stock
            adj = (
                f"My {sig.action} prediction on {summary.ticker} was {outcome_raw}. "
                f"Pattern was: {summary.pattern}. "
                f"In one sentence: what rule should I update?"
            )
            print(col("\n  Self-adjustment:", M))
            for tok in llm.chat(adj):
                print(tok, end="", flush=True)
            print(f"\n  {col('Saved: '+tid, DM)}\n")
        else:
            print(col("  Skipped.", DM))

    print(col("  Training mode ended.\n", M))


# ─────────────────────────────────────────────────────────────────────────────
#  Alert trigger callback
# ─────────────────────────────────────────────────────────────────────────────

def on_alert_trigger(alert, current_price: float):
    _beep()
    print(f"\n  {col('🔔 PRICE ALERT TRIGGERED', Y+BLD)}")
    print(f"  {alert.ticker.replace('.NS','')} is now {col('₹'+str(current_price), W)}  "
          f"({col(alert.direction.upper(), W)} {col('₹'+str(alert.price), Y)})")
    if alert.note:
        print(f"  Note: {alert.note}")
    print(f"  Alert ID: {alert.id}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  REPL
# ─────────────────────────────────────────────────────────────────────────────

def repl(args, analyzer, brain, llm, mem, tracker,
         news_fetcher, alert_mgr, watcher, state):
    live_feed = state.get("live_feed", LIVE_FEED_DEFAULT)
    last_tid  = ""

    print_banner(live_feed)

    if live_feed:
        print(col("  Fetching market news...", DM), end="", flush=True)
        try:
            sentiment = news_fetcher.fetch(force=True)
            print(col(" done", G))
            print(col(f"  Market: {sentiment.overall_signal.upper()} | {sentiment.summary}", DM))
        except Exception:
            print(col(" failed (check internet)", Y))
        tracker.start()
        alert_mgr.start()
    else:
        print(col("  ○ Live feed OFF — analysis uses chart + rulebook only", Y))

    print()

    # Start watcher if configured
    watch_folder = state.get("watch_folder", "")
    if args.watch:
        watch_folder = args.watch
        state["watch_folder"] = watch_folder
        save_state(state)
    if watch_folder:
        watcher.set_folder(watch_folder)
        watcher.start()
        print(col(f"  👁 Watching: {watch_folder}", G))

    if args.image:
        last_tid = run_analysis(
            args.image, analyzer, brain, llm, mem,
            tracker, news_fetcher, alert_mgr, live_feed
        )

    print(col("  Type 'help' for all commands.\n", DM))

    while True:
        try:
            user = input(col("  you › ", C)).strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not user:
            continue
        low = user.lower()

        # ── Commands ──────────────────────────────────────────────────────────

        if low in ("quit","exit","q"):
            break

        elif low in ("help","h","?"):
            print_help()

        elif low == "clear":
            print_banner(live_feed)

        elif low == "reset":
            llm.reset()
            print(col("  Conversation memory cleared.", DM))

        elif low == "reload":
            brain.reload_rulebook()
            print(col("  Rulebook reloaded.", G))

        elif low.startswith("livefeed"):
            parts = low.split()
            toggle = parts[1] if len(parts) > 1 else ""
            if toggle == "on":
                live_feed = True
                tracker.start()
                alert_mgr.start()
                print(col("  ● Live feed ON", G))
                try:
                    news_fetcher.fetch(force=True)
                    print(col("  News fetched.", G))
                except Exception:
                    print(col("  News unavailable.", Y))
            elif toggle == "off":
                live_feed = False
                tracker.stop()
                alert_mgr.stop()
                print(col("  ○ Live feed OFF", Y))
            else:
                status = col("ON", G) if live_feed else col("OFF", Y)
                print(f"  Live feed is currently {status}")
            state["live_feed"] = live_feed
            save_state(state)

        elif low.startswith("watch"):
            parts = user.split(None, 2)
            if len(parts) < 2:
                status = col("ON", G) if watcher._running else col("OFF", Y)
                print(f"  Folder watcher: {status}")
                if watcher._folder:
                    print(f"  Watching: {watcher._folder}")
            elif parts[1].lower() == "off":
                watcher.stop()
                print(col("  Folder watcher stopped.", Y))
            elif parts[1].lower() == "on":
                folder = parts[2] if len(parts) > 2 else state.get("watch_folder", WATCH_FOLDER)
                watcher.set_folder(folder)
                watcher.start()
                state["watch_folder"] = folder
                save_state(state)
                print(col(f"  👁 Watching: {folder}", G))
            else:
                # treat as folder path
                folder = parts[1]
                watcher.set_folder(folder)
                watcher.start()
                state["watch_folder"] = folder
                save_state(state)
                print(col(f"  👁 Watching: {folder}", G))

        elif low.startswith("alert"):
            parts = user.split()
            if len(parts) == 1 or parts[1].lower() in ("list","show",""):
                print_alerts(alert_mgr)
            elif parts[1].lower() == "clear":
                alert_mgr.clear()
                print(col("  All alerts cleared.", DM))
            elif parts[1].lower() == "remove" and len(parts) > 2:
                ok = alert_mgr.remove(parts[2])
                print(col("  Removed." if ok else "  ID not found.", G if ok else R))
            elif len(parts) >= 4:
                # alert ICICI above 1320 [note]
                ticker  = parts[1].upper()
                direc   = parts[2].lower()
                try:
                    price = float(parts[3].replace("₹","").replace(",",""))
                except ValueError:
                    print(col("  Usage: alert ICICI above 1320", Y))
                    continue
                note = " ".join(parts[4:]) if len(parts) > 4 else ""
                if not live_feed:
                    print(col("  Live feed is OFF — alerts need live feed on", Y))
                    continue
                aid = alert_mgr.add(ticker, direc, price, note)
                print(col(f"  Alert set: {aid}  {ticker} {direc} ₹{price}", G))
            else:
                print(col("  Usage: alert <STOCK> above/below <price>", Y))

        elif low == "alerts":
            print_alerts(alert_mgr)

        elif low.startswith("load "):
            path = user[5:].strip().strip('"').strip("'")
            last_tid = run_analysis(
                path, analyzer, brain, llm, mem,
                tracker, news_fetcher, alert_mgr, live_feed
            )

        elif low == "prices":
            if not live_feed:
                print(col("  Live feed is OFF. Type 'livefeed on' first.", Y))
            else:
                snap = tracker.snapshot()
                if not snap:
                    print(col("  Fetching... wait 10s and try again.", DM))
                else:
                    print_live_snapshot(tracker)

        elif low in ("news","sentiment"):
            if not live_feed:
                print(col("  Live feed is OFF.", Y))
            else:
                try:
                    print_news(news_fetcher.fetch())
                except Exception:
                    print(col("  Could not fetch news.", R))

        elif low in ("stats","dashboard","performance"):
            print_stats(mem)

        elif low == "pending":
            trades = mem.pending_trades()
            if not trades:
                print(col("  No pending trades.", DM))
            else:
                print(f"\n  {col('Pending trades:', BLD)}")
                for t in trades:
                    print(f"    {col(t['id'],C)}  {t['stock']:<18} {t['action']}  "
                          f"entry=₹{t['entry']}  date={t['date']}")

        elif low.startswith("outcome "):
            parts = user.split()
            if len(parts) < 3:
                print(col("  Usage: outcome <id> win/loss [exit_price]", Y))
                continue
            tid     = parts[1]
            outcome = parts[2].lower()
            exit_px = float(parts[3].replace("₹","")) if len(parts) > 3 else 0.0
            ok      = mem.record_outcome(tid, outcome, exit_px)
            if ok:
                c_ = G if outcome=="win" else R
                print(col(f"  Outcome recorded: {outcome.upper()}", c_))
                print(col("  Stock profile updated in rulebook.json", DM))
            else:
                print(col(f"  Trade ID '{tid}' not found.", R))

        elif low.startswith("history "):
            stock  = user[8:].strip().upper()
            trades = mem.recall(stock, limit=10)
            if not trades:
                print(col(f"  No history for {stock}", DM))
            else:
                print(f"\n  {col('History: '+stock, BLD)}")
                for t in trades:
                    oc = G if t["outcome"]=="win" else (R if t["outcome"]=="loss" else Y)
                    pnl = f"  pnl=₹{t['pnl']}" if t.get("pnl") else ""
                    print(f"    {t['date']}  {t['action']}  ₹{t['entry']}→₹{t.get('exit_price',0)}  "
                          f"{col(t['outcome'].upper(), oc)}{pnl}")

        elif low in ("train","training"):
            training_mode(analyzer, brain, llm, mem, tracker, live_feed)

        elif low.startswith("backtest"):
            parts = user.split()
            bt_ticker = (parts[1].upper() + ".NS" if not parts[1].upper().endswith(".NS") else parts[1].upper()) if len(parts) > 1 else "RELIANCE.NS"
            print(col(f"\n  Running Walk-Forward Backtest on {bt_ticker}...", C))
            try:
                bt_df = tracker.fetch_candles(bt_ticker, interval="5m", period="1mo")
                if bt_df is None or len(bt_df) < 100:
                    bt_df = tracker.fetch_candles(bt_ticker, interval="5m", period="5d")
                engine = WalkForwardBacktest()
                rpt = engine.run(bt_df, ticker=bt_ticker, warmup_bars=60)
                print_backtest_report(rpt, bt_ticker)
            except Exception as e:
                print(col(f"  Backtest failed: {e}", R))

        elif low == "drift":
            events = brain.feature_store.get_drift_events(limit=10)
            if not events:
                print(col("  No concept drift events recorded. Model is stable.", G))
            else:
                print(f"\n{hr()}")
                print(f"  {col('ADWIN CONCEPT DRIFT AUDIT TRAIL', BLD)}")
                print(hr("─"))
                for ev in events:
                    print(f"  {col(ev['timestamp'][:19], DM)}  {col(ev['metric_name'], Y)}  "
                          f"{ev['value_before']:.3f} → {ev['value_after']:.3f}")
                    print(f"    {ev['message']}")
                print(hr())

        elif low == "retrain":
            print(col("  Retraining PyTorch LSTM model from FeatureStore...", DM))
            res = brain.dl_retrainer.retrain_from_store(min_samples=25, epochs=3)
            if res.get("status") == "completed":
                print(col(f"  Retraining complete: {res['samples_trained']} samples, Loss={res['avg_loss']}", G))
            else:
                print(col(f"  Retraining skipped: {res.get('reason', 'unknown')}", Y))

        elif low in ("news_scan", "scrape_news"):
            print(col("  Running 4x daily FinBERT news scrape cycle...", DM))
            try:
                from news.rss_scraper import RSSNewsScraper
                sc = RSSNewsScraper(feature_store=brain.feature_store)
                res = sc.scrape_and_process()
                print(col(f"  Scrape finished: {res['new_articles_count']} new articles, "
                          f"{res['tickers_updated']} tickers updated.", G))
            except Exception as e:
                print(col(f"  Scrape failed: {e}", R))

        elif low.startswith("model "):
            llm.model = user[6:].strip()
            print(col(f"  Model: {llm.model}", G))

        else:
            # Conversational — uses current stock context (last reset was on last analysis)
            print(col("\n  ARGUS:\n", BLD))
            for tok in llm.chat(user):
                print(tok, end="", flush=True)
            print("\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ARGUS V5 — ML/DL Trading Intelligence")
    parser.add_argument("image",       nargs="?", default=None)
    parser.add_argument("--no-live",   action="store_true")
    parser.add_argument("--train",     action="store_true")
    parser.add_argument("--backtest",  action="store_true", help="Run walk-forward validation")
    parser.add_argument("--watch",     default=None, metavar="FOLDER")
    parser.add_argument("--model",     default=None)
    args = parser.parse_args()

    state = load_state()
    if args.no_live:
        state["live_feed"] = False
        save_state(state)

    analyzer     = ChartAnalyzer()
    brain        = ArgusBrain()
    llm          = OllamaClient(model=args.model or OLLAMA_MODEL)
    tracker      = PriceTracker()
    news_fetcher = NewsFetcher()
    mem          = ArgusMemory()
    alert_mgr    = AlertManager(tracker, on_alert_trigger)

    def _auto_analyze(path):
        """Called by folder watcher when new chart detected."""
        print(col(f"\n  👁 New chart detected: {os.path.basename(path)}", C))
        run_analysis(path, analyzer, brain, llm, mem,
                     tracker, news_fetcher, alert_mgr,
                     state.get("live_feed", True))
        print(col("  you › ", C), end="", flush=True)

    watcher = FolderWatcher(
        folder=state.get("watch_folder", WATCH_FOLDER),
        callback=_auto_analyze,
    )

    if args.backtest:
        from evaluation.backtest import WalkForwardBacktest, print_backtest_report
        import yfinance as yf
        ticker = "RELIANCE.NS"
        print(col(f"Running ARGUS V5 Walk-Forward Backtest on {ticker}...", C))
        df = yf.download(ticker, period="1mo", interval="5m", progress=False, auto_adjust=True)
        if df is None or len(df) < 100:
            df = yf.download(ticker, period="5d", interval="5m", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        engine = WalkForwardBacktest()
        rpt = engine.run(df, ticker=ticker, warmup_bars=60)
        print_backtest_report(rpt, ticker)
        return

    if args.train:
        print_banner(False)
        training_mode(analyzer, brain, llm, mem, tracker, state.get("live_feed", True))
    else:
        repl(args, analyzer, brain, llm, mem,
             tracker, news_fetcher, alert_mgr, watcher, state)

    print(col("\n  ARGUS offline.\n", DM))


if __name__ == "__main__":
    main()
