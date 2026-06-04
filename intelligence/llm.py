"""
ARGUS V3 — LLM Client (Gemma 4)
Streaming Ollama client with intelligent prompt construction.
"""
import json
import requests
from typing import Generator
from config import OLLAMA_URL, OLLAMA_MODEL


SYSTEM_PROMPT = """You are ARGUS, an elite NSE intraday trading analyst with 15+ years experience.

You receive:
1. Chart data extracted by computer vision
2. Today's market news sentiment
3. Past trade history for this stock
4. A pre-computed signal from the intelligence engine

Your job is to reason about ALL of this and give a clear, actionable response.

Rules:
- Be direct. Say BUY, SELL, or WAIT clearly.
- Give exact price levels when possible.
- Always mention the stop loss.
- Explain WHY in simple terms — like talking to a smart friend.
- Mention news impact if relevant.
- Reference past trade history if available.
- Never use markdown formatting. No **, ##, or * symbols. Plain text only.
- This is educational analysis only — NOT financial advice.
- Capital available: ₹4,00,000. Max risk per trade: ₹4,000 (1%).
- Intraday only — all positions must close by 3:15 PM."""


def autodetect_model(base_url: str = OLLAMA_URL) -> str:
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if not models:
            return OLLAMA_MODEL
        for m in models:
            if "gemma4" in m.lower() or "gemma3" in m.lower():
                return m
        return models[0]
    except Exception:
        return OLLAMA_MODEL


class OllamaClient:
    def __init__(self, model: str = None, base_url: str = OLLAMA_URL):
        self.model    = model or OLLAMA_MODEL
        self.base_url = base_url.rstrip("/")

    def chat(self, messages: list, stream: bool = True, temp: float = 0.3):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": temp, "num_predict": 1000, "top_p": 0.9},
        }
        if stream:
            return self._stream(payload)
        return self._blocking(payload)

    def _blocking(self, payload) -> str:
        try:
            r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=120)
            if r.status_code == 404:
                return self._model_err()
            r.raise_for_status()
            return r.json().get("message", {}).get("content", "")
        except requests.exceptions.ConnectionError:
            return self._conn_err()
        except Exception as e:
            return f"[LLM error] {e}"

    def _stream(self, payload) -> Generator:
        try:
            with requests.post(
                f"{self.base_url}/api/chat", json=payload, stream=True, timeout=180
            ) as r:
                if r.status_code == 404:
                    yield self._model_err(); return
                r.raise_for_status()
                for line in r.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if chunk.get("done"):
                            break
        except requests.exceptions.ConnectionError:
            yield self._conn_err()
        except Exception as e:
            yield f"[LLM error] {e}"

    def list_models(self) -> list:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=8)
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []

    def _conn_err(self):
        return "\n[ARGUS] Ollama not running. Start it: ollama serve\n"

    def _model_err(self):
        av = self.list_models()
        hint = "  Available: " + ", ".join(av) if av else ""
        return f"\n[ARGUS] Model '{self.model}' not found.\n{hint}\n"


class ReasoningEngine:
    """Builds prompts from all intelligence layers and streams LLM responses."""

    def __init__(self, client: OllamaClient = None):
        self.client  = client or OllamaClient()
        self._memory: list = []

    def analyze(
        self,
        intel_signal,       # IntelligentSignal
        chart_summary,      # MarketSummary
        news_sentiment,     # MarketSentiment
        memory_context: str,# formatted past trades
        question: str,
        stream: bool = True,
    ):
        context = self._build_context(intel_signal, chart_summary, news_sentiment, memory_context)
        user_msg = {
            "role": "user",
            "content": f"{context}\n\nQuestion: {question}"
        }
        self._memory = [user_msg]
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, user_msg]
        return self.client.chat(messages, stream=stream)

    def followup(self, question: str, stream: bool = True):
        if not self._memory:
            return iter(["Load a chart first."])
        self._memory.append({"role": "user", "content": question})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self._memory
        resp = self.client.chat(messages, stream=stream)
        return resp

    def reset(self):
        self._memory = []

    def _build_context(self, sig, summary, news, mem_ctx) -> str:
        lines = [
            "=== ARGUS INTELLIGENCE REPORT ===",
            "",
            f"Stock:      {summary.ticker or 'Unknown'}",
            f"Timeframe:  {summary.timeframe or 'Unknown'}",
            f"Confidence: {summary.confidence:.0%}",
            "",
            "--- SIGNAL ---",
            f"Action:     {sig.action}  ({sig.conviction} conviction)",
            f"Risk:       {sig.risk}",
            f"Entry:      {sig.entry_price or 'N/A'}",
            f"Stop loss:  {sig.stop_price or 'N/A'}",
            f"Target:     {sig.target_price or 'N/A'}",
            f"Quantity:   {sig.quantity or 'N/A'} shares",
            f"Capital:    ₹{sig.capital_required or 'N/A'}",
            f"Max loss:   ₹{sig.max_loss or 'N/A'}",
            f"R/R ratio:  {sig.rr_ratio or 'N/A'}",
            "",
            "--- SCORES ---",
            f"Chart score:    {sig.chart_score:+.2f}",
            f"News score:     {sig.news_score:+.2f}",
            f"Memory score:   {sig.memory_score:+.2f}",
            f"Combined score: {sig.combined_score:+.2f}",
            "",
            "--- CHART ---",
            f"Trend:      {summary.trend} ({summary.strength})",
            f"Pattern:    {summary.pattern or 'None'}",
            f"Momentum:   {summary.momentum}",
            f"RSI:        {summary.rsi or 'N/A'} ({summary.rsi_state})",
            f"EMA:        {summary.ema_alignment}",
            f"Volume:     {summary.volume}",
            f"Volatility: {summary.volatility}",
            f"Support:    {summary.support or 'N/A'}",
            f"Resistance: {summary.resistance or 'N/A'}",
            "",
            "--- NEWS SENTIMENT ---",
        ]
        if news:
            lines += [
                f"Market mood: {news.overall_signal} ({news.overall_score:+.2f})",
                f"Summary:     {news.summary}",
            ]
            for r in sig.news_reasons:
                lines.append(f"  {r}")
        else:
            lines.append("  News unavailable.")

        lines += ["", "--- MEMORY / HISTORY ---"]
        lines.append(mem_ctx)

        lines += ["", "--- WARNINGS ---"]
        for w in sig.warnings:
            lines.append(f"  ! {w}")

        lines += ["", "--- FINAL ADVICE ---", sig.final_advice]
        return "\n".join(lines)
