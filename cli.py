"""
ARGUS — Terminal CLI
Run: python cli.py path/to/chart.png
"""
import sys
import os
import time
import textwrap
from pathlib import Path

# ── allow imports from project root ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import cv2

from argus_core import ArgusOrchestrator


# ─────────────────────────────────────────────────────────────────────────────
# Terminal color helpers (no external deps — pure ANSI)
# ─────────────────────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

BLACK   = "\033[30m"
RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"
WHITE   = "\033[37m"

BG_BLACK  = "\033[40m"
BG_RED    = "\033[41m"
BG_GREEN  = "\033[42m"
BG_YELLOW = "\033[43m"

def c(text, *codes):
    return "".join(codes) + str(text) + RESET

def buy_badge():   return c(" BUY  ", BOLD, BG_GREEN,  BLACK)
def sell_badge():  return c(" SELL ", BOLD, BG_RED,    WHITE)
def wait_badge():  return c(" WAIT ", BOLD, BG_YELLOW, BLACK)

def signal_badge(action):
    return {"BUY": buy_badge(), "SELL": sell_badge()}.get(action, wait_badge())

def trend_color(val):
    if val in ("bullish", "strong", "increasing", "high"):
        return c(val, GREEN)
    if val in ("bearish", "weak", "decreasing", "very_high"):
        return c(val, RED)
    return c(val, YELLOW)


# ─────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ─────────────────────────────────────────────────────────────────────────────

TERM_WIDTH = min(os.get_terminal_size().columns if hasattr(os, "get_terminal_size") else 80, 100)

def hr(char="─"):
    print(c(char * TERM_WIDTH, DIM))

def header(title):
    print()
    hr("═")
    print(c(f"  {title}", BOLD, CYAN))
    hr("═")

def section(title):
    print()
    print(c(f"  ▸ {title}", BOLD, WHITE))
    hr()

def row(label, value, width=22):
    label_str = c(f"  {label:<{width}}", DIM)
    print(f"{label_str}{value}")

def wrap_print(text, indent=4, width=None):
    w = (width or TERM_WIDTH) - indent
    lines = text.strip().splitlines()
    for line in lines:
        if line.strip() == "":
            print()
            continue
        for wrapped in textwrap.wrap(line, width=w) or [""]:
            print(" " * indent + wrapped)


# ─────────────────────────────────────────────────────────────────────────────
# Display functions
# ─────────────────────────────────────────────────────────────────────────────

def print_banner():
    banner = r"""
    ___    ____  ______  __  _______
   /   |  / __ \/ ____/ / / / / ___/
  / /| | / /_/ / / __  / / / /\__ \ 
 / ___ |/ _, _/ /_/ / / /_/ /___/ / 
/_/  |_/_/ |_|\____/  \____//____/  
"""
    print(c(banner, CYAN, BOLD))
    print(c("  AI Visual Market Analyst  ·  Local · Offline · Private", DIM))
    print(c("  Type 'help' for commands, 'quit' to exit.\n", DIM))


def print_summary(summary, signal):
    header("MARKET ANALYSIS")

    # Signal badge + conviction
    action = signal.action
    print(f"\n  Signal  {signal_badge(action)}  "
          f"{c(signal.conviction + ' conviction', BOLD)}  ·  "
          f"Risk: {trend_color(signal.risk_rating)}\n")

    section("Chart Structure")
    row("Ticker",       c(summary.ticker or "Unknown", BOLD))
    row("Timeframe",    summary.timeframe or "Unknown")
    row("Trend",        trend_color(f"{summary.trend} ({summary.trend_strength})"))
    row("Pattern",      c(summary.chart_pattern or "None detected", YELLOW) if summary.chart_pattern else c("None detected", DIM))
    row("Momentum",     trend_color(summary.momentum))
    row("Volatility",   trend_color(summary.volatility))
    row("Consolidation",c("Yes", YELLOW) if summary.consolidation else c("No", DIM))

    section("Key Levels")
    row("Support",      c(f"{summary.support:.2f}", GREEN)     if summary.support    else c("—", DIM))
    row("Resistance",   c(f"{summary.resistance:.2f}", RED)    if summary.resistance else c("—", DIM))

    section("Indicators")
    rsi_str = f"{summary.rsi:.1f}  ({summary.rsi_state})" if summary.rsi else "Not extracted"
    row("RSI",          trend_color(rsi_str))
    row("EMA Alignment",trend_color(summary.ema_alignment))
    row("MACD State",   c(summary.macd_state.replace("_", " "), CYAN))
    row("Volume",       trend_color(summary.volume_behavior))
    row("Breakout Prob.",trend_color(summary.breakout_probability))

    if signal.entry_zone:
        section("Trade Zones")
        row("Entry",    c(signal.entry_zone,    CYAN))
        row("Stop Loss",c(signal.stop_loss_zone or "—", RED))
        row("Target",   c(signal.target_zone    or "—", GREEN))
        if signal.risk_reward_ratio:
            row("R/R Ratio", c(f"{signal.risk_reward_ratio:.1f}x", BOLD, YELLOW))

    section("Reasoning")
    for pt in signal.reasoning_points:
        print(c(f"  • ", DIM) + pt)

    if signal.warnings:
        section("Warnings")
        for w in signal.warnings:
            print(c(f"  ⚠  ", YELLOW) + c(w, YELLOW))

    conf = summary.confidence
    conf_color = GREEN if conf > 0.6 else YELLOW if conf > 0.35 else RED
    print()
    print(c(f"  Extraction confidence: ", DIM) + c(f"{conf:.0%}", conf_color))
    hr()


def stream_response(token_gen):
    """Print LLM tokens as they arrive, then newline."""
    print()
    print(c("  ARGUS  ", BOLD, BG_BLACK, CYAN), end="  ", flush=True)
    for token in token_gen:
        # Wrap at terminal width
        print(token, end="", flush=True)
    print("\n")


def print_help():
    print()
    print(c("  Commands", BOLD))
    hr()
    cmds = [
        ("load <path>",  "Load a new chart image"),
        ("scan",         "Re-run vision pipeline on current chart"),
        ("summary",      "Re-print the last analysis panel"),
        ("clear",        "Clear the screen"),
        ("model <name>", "Switch Ollama model (e.g. gemma3:4b)"),
        ("models",       "List available Ollama models"),
        ("help",         "Show this help"),
        ("quit / exit",  "Exit ARGUS"),
    ]
    for cmd, desc in cmds:
        print(f"  {c(cmd, CYAN, BOLD):<30}{c(desc, DIM)}")
    print()
    print(c("  Any other text is sent to the LLM as a follow-up question.", DIM))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main REPL
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray | None:
    img = cv2.imread(path)
    if img is None:
        print(c(f"  ✗ Could not load image: {path}", RED))
        return None
    print(c(f"  ✓ Loaded: {path}  ({img.shape[1]}×{img.shape[0]})", GREEN))
    return img


def run_analysis(argus: ArgusOrchestrator, img: np.ndarray) -> tuple | None:
    print(c("\n  Running vision pipeline…", DIM), end="", flush=True)
    t0 = time.time()
    try:
        summary, signal = argus.extract_only(img)
        elapsed = (time.time() - t0) * 1000
        print(c(f"  done ({elapsed:.0f}ms)", DIM))
        return summary, signal
    except Exception as e:
        print(c(f"\n  ✗ Vision pipeline error: {e}", RED))
        return None


def initial_analysis(argus, img):
    """Run vision + stream the first LLM analysis."""
    result = run_analysis(argus, img)
    if not result:
        return None
    summary, signal = result
    print_summary(summary, signal)

    # Stream initial LLM analysis
    print(c("  Generating analysis…", DIM))
    try:
        token_gen = argus.reasoning.analyze_chart(
            summary, signal,
            "Analyze this chart and give me your full assessment.",
            stream=True,
        )
        stream_response(token_gen)
    except Exception as e:
        print(c(f"  ✗ LLM error: {e}", RED))
        print(c("  Tip: make sure 'ollama serve' is running in another terminal.", YELLOW))

    return summary, signal


def main():
    print_banner()

    # ── parse CLI args ────────────────────────────────────────────────────────
    image_path = None
    model = "gemma3"
    args = sys.argv[1:]

    i = 0
    while i < len(args):
        if args[i] in ("--model", "-m") and i + 1 < len(args):
            model = args[i + 1]
            i += 2
        elif not args[i].startswith("-"):
            image_path = args[i]
            i += 1
        else:
            i += 1

    argus = ArgusOrchestrator(ollama_model=model)
    print(c(f"  Model: {model}  ·  Ollama: http://localhost:11434", DIM))

    img = None
    last_result = None

    # ── optional: analyze chart passed as CLI arg ─────────────────────────────
    if image_path:
        img = load_image(image_path)
        if img is not None:
            argus.reset()
            last_result = initial_analysis(argus, img)

    # ── REPL ──────────────────────────────────────────────────────────────────
    while True:
        try:
            user_input = input(c("  you › ", BOLD, CYAN)).strip()
        except (EOFError, KeyboardInterrupt):
            print(c("\n  Goodbye.\n", DIM))
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        # ── built-in commands ─────────────────────────────────────────────────
        if cmd in ("quit", "exit", "q"):
            print(c("\n  Goodbye.\n", DIM))
            break

        elif cmd == "help":
            print_help()

        elif cmd == "clear":
            os.system("cls" if os.name == "nt" else "clear")
            print_banner()

        elif cmd == "summary":
            if last_result:
                print_summary(*last_result)
            else:
                print(c("  No analysis yet. Load a chart first.", YELLOW))

        elif cmd == "scan":
            if img is None:
                print(c("  No chart loaded. Use: load <path>", YELLOW))
            else:
                argus.reset()
                last_result = initial_analysis(argus, img)

        elif cmd == "models":
            models = argus.list_available_models()
            if models:
                print(c("\n  Available models:", BOLD))
                for m in models:
                    marker = c(" ◀ active", GREEN) if m.startswith(model) else ""
                    print(f"    {c(m, CYAN)}{marker}")
                print()
            else:
                print(c("  Could not reach Ollama. Is 'ollama serve' running?", YELLOW))

        elif cmd.startswith("model "):
            new_model = user_input[6:].strip()
            argus.switch_model(new_model)
            model = new_model
            print(c(f"  Switched to model: {new_model}", GREEN))

        elif cmd.startswith("load "):
            path = user_input[5:].strip().strip('"').strip("'")
            new_img = load_image(path)
            if new_img is not None:
                img = new_img
                argus.reset()
                last_result = initial_analysis(argus, img)

        # ── conversational follow-up ──────────────────────────────────────────
        else:
            if last_result is None and img is None:
                print(c("  Load a chart first: load <path>  or  argus cli.py chart.png", YELLOW))
                continue

            if last_result is None and img is not None:
                # image loaded but not yet analyzed
                argus.reset()
                last_result = initial_analysis(argus, img)
                if not last_result:
                    continue

            try:
                token_gen = argus.chat(user_input, stream=True)
                stream_response(token_gen)
            except Exception as e:
                print(c(f"  ✗ {e}", RED))


if __name__ == "__main__":
    main()
