"""
ARGUS V4 — Jarvis Terminal
NSE intraday intelligence with live feed toggle, training mode, and persistent memory.

Usage:
  python argus.py                    → start clean
  python argus.py chart.png          → start with immediate analysis
  python argus.py --no-live          → start with live feed OFF
  python argus.py --train            → start in training mode
"""

import sys, os, json, argparse, time, threading
from datetime import datetime

# ── make imports work from argus_v4/ directory ────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import BANNER, CAPITAL, MAX_RISK_PCT, STATE_PATH, LIVE_FEED_DEFAULT, OLLAMA_MODEL
from vision.analyzer import ChartAnalyzer
from intelligence.brain import ArgusBrain
from intelligence.llm import OllamaClient
from intelligence.price_tracker import PriceTracker
from news.fetcher import NewsFetcher
from memory.store import ArgusMemory, TradeRecord, new_trade_id

# ── ANSI colours ──────────────────────────────────────────────────────────────
R  = "\033[91m"   # red
G  = "\033[92m"   # green
Y  = "\033[93m"   # yellow
B  = "\033[94m"   # blue
C  = "\033[96m"   # cyan
M  = "\033[95m"   # magenta
W  = "\033[97m"   # white
DM = "\033[2m"    # dim
RS = "\033[0m"    # reset
BLD= "\033[1m"    # bold

def c(text, colour=W): return f"{colour}{text}{RS}"
def hr(ch="─", n=58):  return c(ch * n, DM)


# ─────────────────────────────────────────────────────────────────────────────
#  State (persisted across sessions)
# ─────────────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    try:
        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
        with open(STATE_PATH) as f:
            return json.load(f)
    except Exception:
        return {"live_feed": LIVE_FEED_DEFAULT}

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
    print(c(BANNER, C))
    status = c("● LIVE", G) if live_feed else c("○ OFFLINE", Y)
    print(f"  Status: {status}  |  Capital: {c('₹'+format(CAPITAL,','), W)}  |  Risk/trade: {c('1%', Y)}")
    print(hr())

def print_signal(sig, summary=None, stock_acc=None):
    colour = G if sig.action == "BUY" else (R if sig.action == "SELL" else Y)
    print(f"\n  {hr()}")
    print(f"  {c('ARGUS SIGNAL', BLD)}")
    print(hr())
    print(f"  Action    : {c(sig.action, colour+BLD)}  |  Grade: {c(sig.grade, W)}  |  Confidence: {c(str(sig.confidence)+'%', W)}")
    print(f"  Conviction: {sig.conviction}  |  R/R: {c(sig.rr_ratio or 'N/A', W)}")
    print()

    if sig.live_price:
        chg_col = G if sig.live_change_pct >= 0 else R
        print(f"  Live price: {c('₹'+str(sig.live_price), W)}  "
              f"({c(('+' if sig.live_change_pct>=0 else '')+str(sig.live_change_pct)+'%', chg_col)})")

    print(f"  Entry     : {c('₹'+str(sig.entry_price) if sig.entry_price else 'TBD', W)}")
    print(f"  Stop loss : {c('₹'+str(sig.stop_price)  if sig.stop_price  else 'TBD', R)}")
    print(f"  Target    : {c('₹'+str(sig.target_price) if sig.target_price else 'TBD', G)}")
    print()
    print(f"  Quantity  : {c(str(sig.quantity)+' shares', W)}")
    print(f"  Capital   : {c('₹'+format(int(sig.capital_required),',') if sig.capital_required else 'N/A', W)}")
    print(f"  Max loss  : {c('₹'+format(int(sig.max_loss),',') if sig.max_loss else 'N/A', R)}")
    print()
    print(f"  Scores    : chart={c(f'{sig.chart_score:+.2f}',W)}  "
          f"news={c(f'{sig.news_score:+.2f}',W)}  "
          f"memory={c(f'{sig.memory_score:+.2f}',W)}  "
          f"combined={c(f'{sig.combined_score:+.2f}',W)}")

    if stock_acc and stock_acc.get("samples", 0) > 0:
        print(f"  Memory    : {stock_acc['samples']} trades on this stock — "
              f"win rate {c(str(int(stock_acc['win_rate']*100))+'%', G if stock_acc['win_rate']>=0.5 else R)}")

    if sig.warnings:
        print()
        for w_ in sig.warnings:
            print(f"  {c('⚠ '+w_, Y)}")

    print(f"\n  {hr('─',50)}")
    print(f"  {c(sig.final_advice, DM)}")
    print(hr())

def print_live_snapshot(tracker: PriceTracker):
    snap = tracker.snapshot()
    if not snap:
        print(c("  No live data yet.", DM))
        return
    print(f"\n{hr()}")
    print(f"  {'TICKER':<20} {'PRICE':>10} {'CHG%':>8} {'HIGH':>10} {'LOW':>10}")
    print(hr("─"))
    nifty = snap.pop("^NSEI", None)
    bnf   = snap.pop("^NSEBANK", None)
    if nifty:
        chg_c = G if nifty["change_pct"] >= 0 else R
        print(f"  {c('NIFTY 50',BLD):<28} {c('₹'+str(nifty['price']),W):>10} "
              f"{c(('+' if nifty['change_pct']>=0 else '')+str(nifty['change_pct'])+'%', chg_c):>16}")
    if bnf:
        chg_c = G if bnf["change_pct"] >= 0 else R
        print(f"  {c('BANK NIFTY',BLD):<28} {c('₹'+str(bnf['price']),W):>10} "
              f"{c(('+' if bnf['change_pct']>=0 else '')+str(bnf['change_pct'])+'%', chg_c):>16}")
    print(hr("─"))
    for ticker, data in sorted(snap.items()):
        chg_c = G if data["change_pct"] >= 0 else R
        name  = ticker.replace(".NS","")
        print(f"  {name:<20} {c('₹'+str(data['price']),W):>18} "
              f"{c(('+' if data['change_pct']>=0 else '')+str(data['change_pct'])+'%', chg_c):>14} "
              f"{str(data['high']):>10} {str(data['low']):>10}")
    upd = tracker.last_update()
    if upd:
        print(c(f"\n  Last updated: {upd}", DM))
    print(hr())

def print_stats(mem: ArgusMemory):
    s = mem.stats()
    print(f"\n{hr()}")
    print(f"  {c('ARGUS PERFORMANCE DASHBOARD', BLD)}")
    print(hr("─"))
    print(f"  Total analyses : {s['total_trades']}")
    print(f"  Resolved       : {s['resolved']}")
    wr_col = G if s['win_rate'] >= 0.6 else (Y if s['win_rate'] >= 0.4 else R)
    print(f"  Win rate       : {c(str(int(s['win_rate']*100))+'%', wr_col)}")
    pnl_col = G if s['total_pnl'] >= 0 else R
    print(f"  Total P&L      : {c('₹'+format(int(s['total_pnl']),','), pnl_col)}")
    if s['by_stock']:
        print(f"\n  Per-stock breakdown:")
        for st, d in s['by_stock'].items():
            total = d['wins'] + d['losses']
            wr    = d['wins'] / total if total else 0
            wc    = G if wr >= 0.6 else (Y if wr >= 0.4 else R)
            print(f"    {st:<20} {d['wins']}W / {d['losses']}L  {c(str(int(wr*100))+'%', wc)}")
    print(hr())

def print_news(sentiment):
    print(f"\n{hr()}")
    print(f"  {c('MARKET SENTIMENT', BLD)}  ({c(sentiment.fetched_at, DM)})")
    print(hr("─"))
    sig_col = G if sentiment.overall_signal=="bullish" else (R if sentiment.overall_signal=="bearish" else Y)
    print(f"  Overall: {c(sentiment.overall_signal.upper(), sig_col+BLD)}")
    print()
    for sec, data in sentiment.sectors.items():
        sc = G if data.signal=="bullish" else (R if data.signal=="bearish" else Y)
        bar = ("+" if data.signal=="bullish" else ("-" if data.signal=="bearish" else "~"))
        print(f"  {bar} {sec:<12} {c(data.signal, sc)}")
    print()
    for h_ in sentiment.top_headlines[:4]:
        print(c(f"  • {h_[:80]}", DM))
    print(hr())

def print_help():
    cmds = [
        ("load <path>",              "Analyze a chart image"),
        ("prices",                   "Show live NSE watchlist prices"),
        ("news",                     "Show today's market sentiment"),
        ("livefeed on/off",          "Toggle live price + news feed"),
        ("outcome <id> win/loss [exit_price]", "Record trade outcome"),
        ("pending",                  "List trades awaiting outcome"),
        ("stats",                    "Performance dashboard"),
        ("history <STOCK>",          "Past trades for a stock"),
        ("train",                    "Enter training mode (upload chart + paste outcome)"),
        ("reload",                   "Reload rulebook.json from disk"),
        ("model <name>",             "Switch LLM model"),
        ("reset",                    "Clear conversation memory"),
        ("clear",                    "Clear screen"),
        ("help",                     "Show this help"),
        ("quit",                     "Exit ARGUS"),
    ]
    print(f"\n{hr()}")
    for cmd, desc in cmds:
        print(f"  {c(cmd, C):<45} {c(desc, DM)}")
    print(hr())

def print_live_toggle(live_feed: bool):
    if live_feed:
        print(c("\n  ● Live feed ON — real-time prices + news active", G))
    else:
        print(c("\n  ○ Live feed OFF — analysis uses chart + memory only", Y))
        print(c("    Type 'livefeed on' to re-enable", DM))


# ─────────────────────────────────────────────────────────────────────────────
#  Training mode
# ─────────────────────────────────────────────────────────────────────────────

def training_mode(analyzer: ChartAnalyzer, brain: ArgusBrain, llm: OllamaClient,
                   mem: ArgusMemory, live_feed: bool):
    print(c("\n  ═══ TRAINING MODE ═══", M+BLD))
    print(c("  Upload a chart → ARGUS predicts → you reveal outcome → ARGUS learns", DM))
    print(c("  Type 'done' to exit training mode\n", DM))

    while True:
        path = input(c("  chart path (or 'done'): ", M)).strip().strip('"').strip("'")
        if path.lower() == "done":
            break

        if not os.path.exists(path):
            print(c(f"  File not found: {path}", R))
            continue

        print(c("  Analysing chart...", DM))
        summary = analyzer.analyze(path)
        if not summary:
            print(c("  Could not read image.", R))
            continue

        # Predict
        sig = brain.generate_signal(
            chart_summary=summary,
            news_sentiment=None,
            memory_context={},
            live_price=None,
            live_feed_on=False,
        )
        print_signal(sig, summary)

        # Stream LLM reasoning
        prompt = _build_training_prompt(summary, sig)
        print(c("\n  ARGUS prediction:\n", BLD))
        for tok in llm.chat(prompt):
            print(tok, end="", flush=True)
        print("\n")

        # Outcome
        outcome_raw = input(c("  What happened? (win/loss/skip, or paste outcome): ", M)).strip().lower()
        if outcome_raw in ("win","loss","skip"):
            # Save and update
            tid = new_trade_id()
            rec = TradeRecord(
                id=tid, stock=summary.ticker or "UNKNOWN",
                date=datetime.now().strftime("%Y-%m-%d"),
                action=sig.action, entry=sig.entry_price,
                stop=sig.stop_price, target=sig.target_price,
                confidence=sig.confidence,
                patterns=[summary.pattern],
                news_signal="N/A",
                outcome=outcome_raw,
            )
            mem.save_trade(rec)
            mem.record_outcome(tid, outcome_raw)

            # ARGUS self-adjusts
            adj_prompt = (
                f"My prediction was {sig.action} with {sig.confidence}% confidence. "
                f"The pattern was: {summary.pattern}. "
                f"Outcome: {outcome_raw}. "
                f"In 2 sentences: what rule should I update?"
            )
            print(c("\n  ARGUS self-adjustment:", M))
            for tok in llm.chat(adj_prompt):
                print(tok, end="", flush=True)
            print("\n")
            print(c(f"  Trade saved: {tid}", DM))
        else:
            print(c("  Outcome skipped.", DM))

    print(c("  Training mode ended.\n", M))


def _build_training_prompt(summary, sig) -> str:
    return (
        f"TRAINING PREDICTION for {summary.ticker or 'unknown stock'}. "
        f"Trend: {summary.trend} ({summary.strength}). "
        f"Pattern: {summary.pattern}. "
        f"Volume: {summary.volume}. RSI: {summary.rsi} ({summary.rsi_state}). "
        f"My signal: {sig.action} — confidence {sig.confidence}%. "
        f"Matched rules: {', '.join(sig.matched_rules) if sig.matched_rules else 'none'}. "
        f"Reasoning in 3 short bullet points. Then state the key risk to this call."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Main analysis flow
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(path, analyzer, brain, llm, mem, tracker, news_fetcher,
                 live_feed, last_tid):
    if not os.path.exists(path):
        print(c(f"  File not found: {path}", R))
        return last_tid

    print(c(f"\n  Analysing {os.path.basename(path)}...", DM))
    summary = analyzer.analyze(path)
    if not summary:
        print(c("  Could not read image.", R))
        return last_tid

    sentiment   = news_fetcher.fetch() if live_feed else None
    live_price  = tracker.get(summary.ticker + ".NS") if (live_feed and summary.ticker) else None
    stock_acc   = mem.stock_accuracy(summary.ticker or "")
    mem_score_d = mem.stock_accuracy(summary.ticker or "")

    sig = brain.generate_signal(
        chart_summary=summary,
        news_sentiment=sentiment,
        memory_context=mem_score_d,
        live_price=live_price,
        live_feed_on=live_feed,
    )

    print_signal(sig, summary, stock_acc)
    if sentiment and live_feed:
        print_news(sentiment)

    # Save to memory
    tid = new_trade_id()
    rec = TradeRecord(
        id=tid, stock=summary.ticker or "UNKNOWN",
        date=datetime.now().strftime("%Y-%m-%d"),
        action=sig.action, entry=sig.entry_price,
        stop=sig.stop_price, target=sig.target_price,
        confidence=sig.confidence,
        patterns=[summary.pattern],
        news_signal=sentiment.overall_signal if sentiment else "N/A",
        outcome="pending",
    )
    mem.save_trade(rec)
    print(c(f"\n  Trade ID: {tid}  (use 'outcome {tid} win/loss' when resolved)", DM))

    # Stream LLM narrative
    prompt = (
        f"Stock: {summary.ticker or 'unknown'} | Trend: {summary.trend} ({summary.strength}) | "
        f"Pattern: {summary.pattern} | Volume: {summary.volume} | RSI: {summary.rsi} ({summary.rsi_state}) | "
        f"Signal: {sig.action} ({sig.conviction}) | Entry: {sig.entry_price} | "
        f"Stop: {sig.stop_price} | Target: {sig.target_price} | "
        f"News: {sentiment.overall_signal if sentiment else 'unavailable'} | "
        f"Warnings: {', '.join(sig.warnings) if sig.warnings else 'none'} | "
        f"Live feed: {'ON' if live_feed else 'OFF'}. "
        f"Give your complete intraday trade assessment in 5-6 sentences. Be direct."
    )
    print(c("\n  ARGUS:\n", BLD))
    for tok in llm.chat(prompt):
        print(tok, end="", flush=True)
    print("\n")

    return tid


# ─────────────────────────────────────────────────────────────────────────────
#  REPL
# ─────────────────────────────────────────────────────────────────────────────

def repl(args, analyzer, brain, llm, mem, tracker, news_fetcher, state):
    live_feed = state.get("live_feed", LIVE_FEED_DEFAULT)
    last_tid  = None

    print_banner(live_feed)
    print_live_toggle(live_feed)

    # Startup: fetch news + start tracker
    if live_feed:
        print(c("  Fetching market news...", DM), end="", flush=True)
        try:
            sentiment = news_fetcher.fetch(force=True)
            print(c(" done", G))
            print(c(f"  Market: {sentiment.overall_signal.upper()} | {sentiment.summary}", DM))
        except Exception:
            print(c(" failed (offline?)", Y))
        tracker.start()
    
    print()
    if args.image:
        last_tid = run_analysis(
            args.image, analyzer, brain, llm, mem,
            tracker, news_fetcher, live_feed, last_tid
        )

    print(c("  Type 'help' for all commands.\n", DM))

    while True:
        try:
            user = input(c("  you › ", C)).strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not user:
            continue

        low = user.lower()

        # ── Commands ──────────────────────────────────────────────────────────

        if low in ("quit", "exit", "q"):
            break

        elif low in ("help", "h", "?"):
            print_help()

        elif low == "clear":
            print_banner(live_feed)
            print_live_toggle(live_feed)

        elif low == "reset":
            llm.reset()
            print(c("  Conversation memory cleared.", DM))

        elif low == "reload":
            brain.reload_rulebook()
            print(c("  Rulebook reloaded from rulebook.json", G))

        elif low.startswith("livefeed"):
            parts = low.split()
            if len(parts) < 2:
                print_live_toggle(live_feed)
                continue
            toggle = parts[1]
            if toggle == "on":
                live_feed = True
                tracker.start()
                print(c("\n  ● Live feed ON", G))
                print(c("  Starting price tracker + fetching news...", DM))
                try:
                    news_fetcher.fetch(force=True)
                    print(c("  News fetched.", G))
                except Exception:
                    print(c("  News fetch failed.", Y))
            elif toggle == "off":
                live_feed = False
                tracker.stop()
                print(c("\n  ○ Live feed OFF — using chart + memory only", Y))
            else:
                print(c("  Usage: livefeed on  |  livefeed off", Y))
            state["live_feed"] = live_feed
            save_state(state)

        elif low.startswith("load "):
            path     = user[5:].strip().strip('"').strip("'")
            last_tid = run_analysis(
                path, analyzer, brain, llm, mem,
                tracker, news_fetcher, live_feed, last_tid
            )

        elif low == "prices":
            if not live_feed:
                print(c("  Live feed is OFF. Turn it on first: livefeed on", Y))
            elif not tracker.snapshot():
                print(c("  Fetching prices (first run, ~15s)...", DM))
                time.sleep(3)
                print_live_snapshot(tracker)
            else:
                print_live_snapshot(tracker)

        elif low in ("news", "sentiment"):
            if not live_feed:
                print(c("  Live feed is OFF. Turn it on first: livefeed on", Y))
            else:
                try:
                    s = news_fetcher.fetch()
                    print_news(s)
                except Exception:
                    print(c("  Could not fetch news.", R))

        elif low in ("stats", "dashboard", "performance"):
            print_stats(mem)

        elif low == "pending":
            trades = mem.pending_trades()
            if not trades:
                print(c("  No pending trades.", DM))
            else:
                print(f"\n  {c('Pending trades:', BLD)}")
                for t in trades:
                    print(f"    {c(t['id'], C)}  {t['stock']}  {t['action']}  entry={t['entry']}  "
                          f"date={t['date']}")

        elif low.startswith("outcome "):
            parts = user.split()
            if len(parts) < 3:
                print(c("  Usage: outcome <id> win/loss [exit_price]", Y))
                continue
            tid     = parts[1]
            outcome = parts[2].lower()
            exit_px = float(parts[3]) if len(parts) > 3 else 0.0
            ok      = mem.record_outcome(tid, outcome, exit_px)
            if ok:
                col = G if outcome == "win" else R
                print(c(f"  Outcome recorded: {outcome.upper()}", col))
                print(c("  Rulebook updated.", DM))
            else:
                print(c(f"  Trade ID {tid} not found.", R))

        elif low.startswith("history "):
            stock  = user[8:].strip().upper()
            trades = mem.recall(stock, limit=10)
            if not trades:
                print(c(f"  No history for {stock}", DM))
            else:
                print(f"\n  {c('History: '+stock, BLD)}")
                for t in trades:
                    oc = G if t['outcome']=="win" else (R if t['outcome']=="loss" else Y)
                    print(f"    {t['date']}  {t['action']}  entry={t['entry']}  "
                          f"outcome={c(t['outcome'].upper(), oc)}"
                          f"{'  pnl=₹'+str(t['pnl']) if t['pnl'] else ''}")

        elif low in ("train", "training"):
            training_mode(analyzer, brain, llm, mem, live_feed)

        elif low.startswith("model "):
            llm.model = user[6:].strip()
            print(c(f"  Model switched to: {llm.model}", G))

        else:
            # Pass-through to LLM
            print(c("\n  ARGUS:\n", BLD))
            for tok in llm.chat(user):
                print(tok, end="", flush=True)
            print("\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ARGUS V4 — NSE Intelligence Terminal")
    parser.add_argument("image",     nargs="?", default=None, help="Chart image to analyse on startup")
    parser.add_argument("--no-live", action="store_true",     help="Start with live feed OFF")
    parser.add_argument("--train",   action="store_true",     help="Start in training mode")
    parser.add_argument("--model",   default=None,            help="Override Ollama model")
    args = parser.parse_args()

    state = load_state()
    if args.no_live:
        state["live_feed"] = False
        save_state(state)

    analyzer    = ChartAnalyzer()
    brain       = ArgusBrain()
    llm         = OllamaClient(model=args.model or OLLAMA_MODEL)
    tracker     = PriceTracker()
    news_fetcher= NewsFetcher()
    mem         = ArgusMemory()

    if args.train:
        print_banner(False)
        training_mode(analyzer, brain, llm, mem, state.get("live_feed", True))
    else:
        repl(args, analyzer, brain, llm, mem, tracker, news_fetcher, state)

    print(c("\n  ARGUS offline.\n", DM))


if __name__ == "__main__":
    main()
