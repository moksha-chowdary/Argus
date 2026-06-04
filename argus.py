"""
ARGUS — AI Visual Market Analyst
Terminal CLI  |  Windows / Linux / macOS
Usage:
    python argus.py                        # interactive, no chart yet
    python argus.py chart.png              # analyze immediately
    python argus.py chart.png --model gemma3:4b
"""
import sys
import os
import time
import textwrap
import argparse
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import cv2

from vision.preprocessor import ImagePreprocessor
from vision.ocr           import ChartOCR
from vision.cv_analyzer   import CVAnalyzer
from intelligence.engine  import SummaryBuilder, SignalGenerator
from conversation.llm     import OllamaClient, ReasoningPipeline


# ─────────────────────────────────────────────────────────────────────────────
#  ANSI colors — work on Windows 10+ (enabled below)
# ─────────────────────────────────────────────────────────────────────────────

def _enable_ansi_windows():
    """Enable VT100 escape codes on Windows 10+."""
    if sys.platform == "win32":
        try:
            import ctypes
            k = ctypes.windll.kernel32
            k.SetConsoleMode(k.GetStdHandle(-11), 7)
        except Exception:
            pass

_enable_ansi_windows()

R  = "\033[0m"
B  = "\033[1m"
D  = "\033[2m"
GR = "\033[32m"
RD = "\033[31m"
YL = "\033[33m"
CY = "\033[36m"
WH = "\033[37m"
MG = "\033[35m"
BG_GR = "\033[42m"
BG_RD = "\033[41m"
BG_YL = "\033[43m"
BK    = "\033[30m"

def c(text, *codes):
    return "".join(codes) + str(text) + R

def _tw():
    try:
        return min(os.get_terminal_size().columns, 100)
    except Exception:
        return 80

TW = _tw()


# ─────────────────────────────────────────────────────────────────────────────
#  UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def hr(ch="─"):
    print(c(ch * TW, D))

def section(title):
    print()
    print(c(f"  ▸ {title}", B, WH))
    hr()

def row(label, val, w=24):
    print(f"{c(f'  {label:<{w}}', D)}{val}")

def wrap(text, indent=4):
    for line in text.strip().splitlines():
        if not line.strip():
            print()
            continue
        for seg in textwrap.wrap(line, TW - indent) or [""]:
            print(" " * indent + seg)

def badge(action):
    if action == "BUY":
        return c("  BUY  ", B, BG_GR, BK)
    if action == "SELL":
        return c(" SELL  ", B, BG_RD, WH)
    return c(" WAIT  ", B, BG_YL, BK)

def tcolor(val):
    pos = {"bullish","strong","increasing","high","above_zero","oversold"}
    neg = {"bearish","weak","decreasing","very_high","overbought","below_zero"}
    if str(val).lower() in pos: return c(val, GR)
    if str(val).lower() in neg: return c(val, RD)
    return c(val, YL)


# ─────────────────────────────────────────────────────────────────────────────
#  Banner
# ─────────────────────────────────────────────────────────────────────────────

BANNER = r"""
    ___    ____  ______  __  _______
   /   |  / __ \/ ____/ / / / / ___/
  / /| | / /_/ / / __  / / / /\__ \
 / ___ |/ _, _/ /_/ / / /_/ /___/ /
/_/  |_/_/ |_|\____/  \____//____/
"""

def print_banner(model):
    print(c(BANNER, CY, B))
    print(c("  AI Visual Market Analyst  ·  Local · Offline · Private", D))
    print(c(f"  Model: {model}  ·  Ollama: http://localhost:11434", D))
    print(c("  Type 'help' for commands.\n", D))


# ─────────────────────────────────────────────────────────────────────────────
#  Analysis display
# ─────────────────────────────────────────────────────────────────────────────

def print_panel(summary, signal):
    print()
    hr("═")
    print(c("  MARKET ANALYSIS", B, CY))
    hr("═")

    conf_col = GR if summary.confidence > 0.6 else YL if summary.confidence > 0.35 else RD
    print(f"\n  {badge(signal.action)}  "
          f"{c(signal.conviction + ' conviction', B)}  ·  "
          f"Risk: {tcolor(signal.risk)}  ·  "
          f"Confidence: {c(f'{summary.confidence:.0%}', conf_col)}\n")

    section("Chart Structure")
    row("Ticker",        c(summary.ticker or "Unknown", B))
    row("Timeframe",     summary.timeframe or "Unknown")
    row("Trend",         tcolor(f"{summary.trend} ({summary.strength})"))
    row("Pattern",       c(summary.pattern.replace("_"," "), YL) if summary.pattern else c("None detected", D))
    row("Momentum",      tcolor(summary.momentum))
    row("Volatility",    tcolor(summary.volatility))
    row("Consolidation", c("Yes", YL) if summary.consolidation else c("No", D))

    section("Key Levels")
    row("Support",    c(f"{summary.support:.2f}", GR)  if summary.support    else c("—", D))
    row("Resistance", c(f"{summary.resistance:.2f}", RD) if summary.resistance else c("—", D))

    section("Indicators")
    rsi_str = f"{summary.rsi:.1f}  ({summary.rsi_state})" if summary.rsi else "Not extracted"
    row("RSI",         tcolor(rsi_str))
    row("EMA",         tcolor(summary.ema_alignment))
    row("MACD",        tcolor(summary.macd_state.replace("_"," ")))
    row("Volume",      tcolor(summary.volume))
    row("Breakout P.", tcolor(summary.breakout_prob))

    if signal.entry:
        section("Trade Zones")
        row("Entry",    c(signal.entry,  CY))
        row("Stop",     c(signal.stop or "—", RD))
        row("Target",   c(signal.target or "—", GR))
        if signal.rr_ratio:
            row("R/R", c(f"{signal.rr_ratio:.1f}x", B, YL))

    section("Reasoning")
    for pt in signal.reasons:
        print(c("  • ", D) + pt)

    section("Warnings")
    for w in signal.warnings:
        print(c("  ⚠  ", YL) + c(w, YL))

    hr()


def stream_llm(token_gen):
    print()
    print(c("  ARGUS", B, BG_YL, BK) + "  ", end="", flush=True)
    buf = ""
    col = 8  # already printed "  ARGUS  " = 8 chars
    for token in token_gen:
        for ch in token:
            if ch == "\n":
                print()
                col = 0
            else:
                if col >= TW - 4:
                    print()
                    print("    ", end="", flush=True)
                    col = 4
                print(ch, end="", flush=True)
                col += 1
                buf += ch
    print("\n")
    return buf


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_vision(img_path_or_array, preprocessor, ocr, cv_analyzer, summary_builder, signal_gen):
    print(c("\n  Running vision pipeline", D), end="", flush=True)
    t0 = time.time()
    try:
        processed = preprocessor.preprocess(img_path_or_array)
        print(c(".", D), end="", flush=True)
        ocr_result = ocr.extract(processed)
        print(c(".", D), end="", flush=True)
        cv_result  = cv_analyzer.analyze(processed)
        print(c(".", D), end="", flush=True)
        summary    = summary_builder.build(cv_result, ocr_result)
        signal     = signal_gen.generate(summary)
        elapsed    = (time.time() - t0) * 1000
        print(c(f" done ({elapsed:.0f}ms)", D))
        return summary, signal
    except Exception as e:
        print(c(f"\n  ✗ Vision error: {e}", RD))
        import traceback; traceback.print_exc()
        return None, None


def load_img(path: str):
    p = path.strip().strip('"').strip("'")
    img = cv2.imread(p)
    if img is None:
        print(c(f"  ✗ Cannot load: {p}", RD))
        return None, None
    print(c(f"  ✓ {p}  ({img.shape[1]}×{img.shape[0]})", GR))
    return img, p


def print_help():
    print()
    print(c("  Commands", B))
    hr()
    cmds = [
        ("load <path>",   "Load a new chart image and analyze it"),
        ("scan",          "Re-run vision on the current chart"),
        ("panel",         "Reprint the analysis panel"),
        ("models",        "List available Ollama models"),
        ("model <name>",  "Switch model (e.g. model gemma3:4b)"),
        ("clear",         "Clear the screen"),
        ("help",          "Show this message"),
        ("quit",          "Exit"),
        ("",              ""),
        ("<any question>","Ask the LLM about the current chart"),
    ]
    for cmd, desc in cmds:
        if cmd == "":
            print()
            continue
        print(f"  {c(cmd, CY, B):<35}{c(desc, D)}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  Main REPL
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ARGUS — AI Visual Market Analyst")
    parser.add_argument("chart",   nargs="?", help="Path to chart screenshot")
    parser.add_argument("--model", "-m", default="gemma3", help="Ollama model name")
    parser.add_argument("--url",   default="http://localhost:11434", help="Ollama base URL")
    args = parser.parse_args()

    model = args.model
    print_banner(model)

    # ── init pipeline ─────────────────────────────────────────────────────────
    preprocessor    = ImagePreprocessor()
    ocr             = ChartOCR()
    cv_analyzer     = CVAnalyzer()
    summary_builder = SummaryBuilder()
    signal_gen      = SignalGenerator()
    llm_client      = OllamaClient(model=model, base_url=args.url)
    reasoning       = ReasoningPipeline(client=llm_client)

    img          = None
    last_result  = None   # (summary, signal)

    # ── optional: analyze chart passed as CLI arg ──────────────────────────────
    if args.chart:
        img, _ = load_img(args.chart)
        if img is not None:
            reasoning.reset()
            summary, signal = run_vision(img, preprocessor, ocr, cv_analyzer,
                                         summary_builder, signal_gen)
            if summary:
                last_result = (summary, signal)
                print_panel(summary, signal)
                print(c("  Generating analysis…", D))
                stream_llm(reasoning.analyze(summary, signal,
                    "Analyze this chart and give me your full assessment.", stream=True))

    # ── REPL ──────────────────────────────────────────────────────────────────
    while True:
        try:
            user = input(c("  you › ", B, CY)).strip()
        except (EOFError, KeyboardInterrupt):
            print(c("\n  Goodbye.\n", D))
            break

        if not user:
            continue

        low = user.lower()

        # ── built-in commands ──────────────────────────────────────────────────
        if low in ("quit", "exit", "q"):
            print(c("\n  Goodbye.\n", D))
            break

        elif low == "help":
            print_help()

        elif low == "clear":
            os.system("cls" if os.name == "nt" else "clear")
            print_banner(model)

        elif low == "panel":
            if last_result:
                print_panel(*last_result)
            else:
                print(c("  No analysis yet. Load a chart first.", YL))

        elif low == "scan":
            if img is None:
                print(c("  No chart loaded. Use: load <path>", YL))
            else:
                reasoning.reset()
                summary, signal = run_vision(img, preprocessor, ocr, cv_analyzer,
                                              summary_builder, signal_gen)
                if summary:
                    last_result = (summary, signal)
                    print_panel(summary, signal)
                    print(c("  Generating analysis…", D))
                    stream_llm(reasoning.analyze(summary, signal,
                        "Analyze this chart.", stream=True))

        elif low == "models":
            ms = llm_client.list_models()
            if ms:
                print(c("\n  Available models:", B))
                for m in ms:
                    mark = c("  ◀ active", GR) if model in m else ""
                    print(f"    {c(m, CY)}{mark}")
                print()
            else:
                print(c("  Cannot reach Ollama. Is 'ollama serve' running?", YL))

        elif low.startswith("model "):
            model = user[6:].strip()
            llm_client.model = model
            print(c(f"  ✓ Switched to: {model}", GR))

        elif low.startswith("load "):
            path = user[5:].strip()
            new_img, _ = load_img(path)
            if new_img is not None:
                img = new_img
                reasoning.reset()
                summary, signal = run_vision(img, preprocessor, ocr, cv_analyzer,
                                              summary_builder, signal_gen)
                if summary:
                    last_result = (summary, signal)
                    print_panel(summary, signal)
                    print(c("  Generating analysis…", D))
                    stream_llm(reasoning.analyze(summary, signal,
                        "Analyze this chart and give me your full assessment.", stream=True))

        # ── LLM follow-up ──────────────────────────────────────────────────────
        else:
            if last_result is None and img is None:
                print(c("  Load a chart first: load chart.png", YL))
                continue
            if last_result is None and img is not None:
                reasoning.reset()
                summary, signal = run_vision(img, preprocessor, ocr, cv_analyzer,
                                              summary_builder, signal_gen)
                if summary:
                    last_result = (summary, signal)
                    print_panel(summary, signal)
            if last_result:
                stream_llm(reasoning.followup(user, stream=True))


if __name__ == "__main__":
    main()
