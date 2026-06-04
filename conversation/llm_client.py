"""
ARGUS Conversation Layer — Gemma LLM Integration
Wraps local Ollama / llama.cpp endpoint with structured prompting
for conversational chart analysis.
"""
import json
import requests
from dataclasses import dataclass, field
from typing import Optional, Generator
from intelligence.market_summary import MarketSummary
from intelligence.signal_generator import TradingSignal


# ─── Ollama client ────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api"
DEFAULT_MODEL = "gemma3"          # Use "gemma3:4b" or "gemma3:12b" as available
FALLBACK_MODELS = ["gemma3:4b", "llama3", "mistral", "phi3"]


class OllamaClient:
    """Thin wrapper around the Ollama HTTP API."""

    def __init__(self, base_url: str = OLLAMA_URL, model: str = DEFAULT_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def chat(
        self,
        messages: list[dict],
        stream: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 800,
    ) -> str | Generator[str, None, None]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
            },
        }
        if stream:
            return self._stream(payload)
        return self._blocking(payload)

    def _blocking(self, payload: dict) -> str:
        try:
            resp = requests.post(f"{self.base_url}/chat", json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")
        except requests.exceptions.ConnectionError:
            return self._connection_error_message()
        except Exception as e:
            return f"[ARGUS Error] LLM call failed: {e}"

    def _stream(self, payload: dict) -> Generator[str, None, None]:
        try:
            with requests.post(
                f"{self.base_url}/chat", json=payload, stream=True, timeout=180
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if chunk.get("done"):
                            break
        except requests.exceptions.ConnectionError:
            yield self._connection_error_message()

    def list_models(self) -> list[str]:
        try:
            resp = requests.get(f"{self.base_url}/tags", timeout=10)
            models = resp.json().get("models", [])
            return [m["name"] for m in models]
        except Exception:
            return []

    def _connection_error_message(self) -> str:
        return (
            "⚠️ Cannot connect to Ollama. Please ensure Ollama is running:\n"
            "  `ollama serve`\n"
            f"Then pull a model: `ollama pull {self.model}`"
        )


# ─── Prompt builder ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are ARGUS, a highly experienced technical market analyst with 15+ years in equities, crypto, and derivatives trading.

You analyze chart screenshots through a computer vision pipeline. The vision system has extracted technical data which you will receive as structured JSON. Your job is to:
1. Reason about the extracted data like a senior human analyst
2. Explain setups clearly and conversationally
3. Give honest BUY / SELL / WAIT recommendations
4. Always communicate uncertainty — you are reading image data, not live feeds
5. Estimate risk/reward honestly
6. Behave like a thoughtful analyst, not a trading bot

IMPORTANT RULES:
- NEVER claim certainty about price movements — markets are probabilistic
- Always mention the confidence level of the visual extraction
- If the data is ambiguous, say so clearly
- Frame recommendations as "setups to watch" not guaranteed outcomes
- This is educational analysis only — NOT financial advice
- Always mention that users should verify with their own research

Keep your tone: analytical, grounded, conversational. Like a respected colleague, not a salesperson."""


class ReasoningPipeline:
    """
    Main reasoning pipeline.
    Takes a MarketSummary + TradingSignal → generates analyst-style response.
    """

    def __init__(self, client: Optional[OllamaClient] = None):
        self.client = client or OllamaClient()
        self._memory: list[dict] = []  # session-level chat history

    def analyze_chart(
        self,
        summary: MarketSummary,
        signal: TradingSignal,
        user_question: str = "Analyze this chart and give me your assessment.",
        stream: bool = True,
    ) -> str | Generator[str, None, None]:
        """
        Primary analysis call — takes vision output and produces
        a full conversational analyst response.
        """
        context = self._build_context(summary, signal)
        messages = self._build_messages(context, user_question)
        response = self.client.chat(messages, stream=stream)

        if not stream:
            self._memory.append({"role": "assistant", "content": response})

        return response

    def follow_up(
        self,
        user_message: str,
        stream: bool = True,
    ) -> str | Generator[str, None, None]:
        """
        Follow-up conversational question in the same chart session.
        Uses in-memory chat history for context.
        """
        if not self._memory:
            return "Please analyze a chart first before asking follow-up questions."

        self._memory.append({"role": "user", "content": user_message})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self._memory
        response = self.client.chat(messages, stream=stream)

        if not stream:
            self._memory.append({"role": "assistant", "content": response})

        return response

    def reset_session(self):
        """Clear chat memory between different chart uploads."""
        self._memory = []

    # ─── Prompt construction ──────────────────────────────────────────────────

    def _build_context(self, summary: MarketSummary, signal: TradingSignal) -> str:
        """Serialize the vision output into a concise context block."""
        context_parts = [
            "=== ARGUS VISION EXTRACTION ===",
            f"Ticker: {summary.ticker or 'Unknown'}",
            f"Timeframe: {summary.timeframe or 'Unknown'}",
            f"Extraction confidence: {summary.confidence:.0%}",
            "",
            "--- Technical State ---",
            f"Trend: {summary.trend} ({summary.trend_strength})",
            f"Chart Pattern: {summary.chart_pattern or 'None detected'}",
            f"Momentum: {summary.momentum}",
            f"Volatility: {summary.volatility}",
            f"Consolidation: {'Yes' if summary.consolidation else 'No'}",
            "",
            "--- Key Levels ---",
            f"Support: {summary.support or 'Not extracted'}",
            f"Resistance: {summary.resistance or 'Not extracted'}",
            "",
            "--- Indicators ---",
            f"RSI: {summary.rsi or 'Not extracted'} ({summary.rsi_state})",
            f"EMA Alignment: {summary.ema_alignment}",
            f"MACD State: {summary.macd_state}",
            "",
            "--- Volume ---",
            f"Volume Behavior: {summary.volume_behavior}",
            "",
            "--- Signal Output ---",
            f"Signal: {signal.action}",
            f"Conviction: {signal.conviction}",
            f"Risk Rating: {signal.risk_rating}",
            f"Risk/Reward Ratio: {signal.risk_reward_ratio or 'N/A'}",
            f"Entry Zone: {signal.entry_zone or 'N/A'}",
            f"Stop Loss Zone: {signal.stop_loss_zone or 'N/A'}",
            f"Target Zone: {signal.target_zone or 'N/A'}",
            f"Breakout Probability: {summary.breakout_probability}",
            "",
            "--- Reasoning Points ---",
        ]
        for pt in signal.reasoning_points:
            context_parts.append(f"• {pt}")

        context_parts.append("")
        context_parts.append("--- Warnings ---")
        for w in signal.warnings:
            context_parts.append(f"⚠ {w}")

        return "\n".join(context_parts)

    def _build_messages(self, context: str, user_question: str) -> list[dict]:
        self._memory = []  # fresh analysis = fresh memory

        system_msg = {"role": "system", "content": SYSTEM_PROMPT}
        context_msg = {
            "role": "user",
            "content": (
                f"Here is the extracted chart data from the vision pipeline:\n\n"
                f"{context}\n\n"
                f"Based on this analysis, please respond to the following:\n{user_question}"
            ),
        }
        self._memory.append(context_msg)
        return [system_msg, context_msg]
