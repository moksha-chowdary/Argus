import json
import requests
from typing import Generator

OLLAMA_URL  = "http://localhost:11434"
DEFAULT_MODEL = "gemma3"   # Ollama tag for Gemma 4 — change if yours differs

SYSTEM_PROMPT = """You are ARGUS, a senior technical market analyst with 15+ years experience.

You receive structured chart data extracted by a computer vision pipeline and reason about it like a real analyst — conversational, honest, and precise.

Rules:
- Never claim certainty. Markets are probabilistic.
- Always communicate confidence level.
- Frame recommendations as setups to watch, not guarantees.
- Be concise but thorough.
- This is educational analysis only — NOT financial advice.
- Think like a trader, talk like a colleague."""


class OllamaClient:
    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = OLLAMA_URL):
        self.model    = model
        self.base_url = base_url.rstrip("/")

    def chat(self, messages: list, stream: bool = True, temperature: float = 0.3) -> str | Generator:
        payload = {
            "model":   self.model,
            "messages": messages,
            "stream":  stream,
            "options": {"temperature": temperature, "num_predict": 900, "top_p": 0.9},
        }
        if stream:
            return self._stream(payload)
        return self._blocking(payload)

    def _blocking(self, payload) -> str:
        try:
            r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=120)
            r.raise_for_status()
            return r.json().get("message", {}).get("content", "")
        except requests.exceptions.ConnectionError:
            return self._err()
        except Exception as e:
            return f"[LLM error] {e}"

    def _stream(self, payload) -> Generator:
        try:
            with requests.post(f"{self.base_url}/api/chat", json=payload,
                               stream=True, timeout=180) as r:
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
            yield self._err()
        except Exception as e:
            yield f"[LLM error] {e}"

    def list_models(self) -> list:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=8)
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []

    def _err(self) -> str:
        return (
            "\n[ARGUS] Cannot reach Ollama.\n"
            "  1. Open a new terminal\n"
            "  2. Run: ollama serve\n"
            f"  3. Make sure model '{self.model}' is pulled: ollama pull {self.model}\n"
        )


class ReasoningPipeline:
    def __init__(self, client: OllamaClient = None):
        self.client  = client or OllamaClient()
        self._memory: list = []

    def analyze(self, summary, signal, question: str, stream=True):
        context = self._build_context(summary, signal)
        user_msg = {
            "role": "user",
            "content": f"Chart data from vision pipeline:\n\n{context}\n\nQuestion: {question}"
        }
        self._memory = [user_msg]
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, user_msg]
        resp = self.client.chat(messages, stream=stream)
        if not stream:
            self._memory.append({"role": "assistant", "content": resp})
        return resp

    def followup(self, question: str, stream=True):
        if not self._memory:
            return iter(["No chart analyzed yet. Load a chart first."])
        self._memory.append({"role": "user", "content": question})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self._memory
        resp = self.client.chat(messages, stream=stream)
        if not stream:
            self._memory.append({"role": "assistant", "content": resp})
        return resp

    def reset(self):
        self._memory = []

    def _build_context(self, s, sig) -> str:
        lines = [
            f"Ticker: {s.ticker or 'Unknown'}  |  Timeframe: {s.timeframe or 'Unknown'}",
            f"Confidence: {s.confidence:.0%}",
            "",
            f"Trend:       {s.trend} ({s.strength})",
            f"Pattern:     {s.pattern or 'None'}",
            f"Momentum:    {s.momentum}",
            f"Volatility:  {s.volatility}",
            f"Consolidation: {'Yes' if s.consolidation else 'No'}",
            "",
            f"Support:     {s.support or 'N/A'}",
            f"Resistance:  {s.resistance or 'N/A'}",
            "",
            f"RSI:         {s.rsi or 'N/A'} ({s.rsi_state})",
            f"EMA:         {s.ema_alignment}",
            f"MACD:        {s.macd_state}",
            f"Volume:      {s.volume}",
            f"Breakout P.: {s.breakout_prob}",
            "",
            f"Signal:      {sig.action}  ({sig.conviction} conviction)",
            f"Risk:        {sig.risk}",
            f"R/R:         {sig.rr_ratio or 'N/A'}",
            f"Entry:       {sig.entry or 'N/A'}",
            f"Stop:        {sig.stop or 'N/A'}",
            f"Target:      {sig.target or 'N/A'}",
            "",
            "Reasoning:",
        ]
        for r in sig.reasons:
            lines.append(f"  • {r}")
        lines.append("\nWarnings:")
        for w in sig.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)
