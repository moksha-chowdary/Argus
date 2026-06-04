"""
ARGUS V4 — LLM Client (Gemma4 via Ollama)
Short prompts. GPU-offloaded. Streams token-by-token.
"""

import requests, json
from typing import Generator
from config import OLLAMA_URL, OLLAMA_MODEL, MAX_TOKENS, CTX_WINDOW, TEMPERATURE


SYSTEM_PROMPT = """You are ARGUS, an elite NSE intraday trading analyst.
Rules:
- Capital ₹4,00,000. Max risk 1% per trade = ₹4,000.
- NSE intraday only. Exit before 3:15 PM.
- Never buy support without higher-low structure or volume confirmation.
- Lower highs = bearish until structure breaks.
- Never short after extended decline without exhaustion check.
- Two 15m candle confirmation required after 9:15 open.
- Plain text only. No markdown, no bullet symbols, no ** or ##.
- Be brutally honest. Short and direct answers only.
- When live feed is OFF, note that analysis lacks real-time price context."""


class OllamaClient:
    def __init__(self, model: str = OLLAMA_MODEL):
        self.model   = model
        self._memory = []  # conversation history

    def chat(self, user_msg: str, stream: bool = True) -> Generator[str, None, None]:
        self._memory.append({"role": "user", "content": user_msg})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self._memory[-8:]

        payload = {
            "model":    self.model,
            "messages": messages,
            "stream":   stream,
            "options":  {
                "temperature":  TEMPERATURE,
                "num_predict":  MAX_TOKENS,
                "num_ctx":      CTX_WINDOW,
                "num_gpu":      99,   # offload all layers to RTX 4050
            },
        }

        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json=payload, stream=stream, timeout=120,
            )
            if r.status_code == 404:
                yield self._model_err()
                return
            r.raise_for_status()

            full = ""
            if stream:
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        tok   = chunk.get("message", {}).get("content", "")
                        if tok:
                            full += tok
                            yield tok
                    except Exception:
                        pass
            else:
                data = r.json()
                full = data.get("message", {}).get("content", "")
                yield full

            if full:
                self._memory.append({"role": "assistant", "content": full})

        except requests.ConnectionError:
            yield "\n[ARGUS] Cannot reach Ollama. Run: ollama serve\n"
        except Exception as e:
            yield f"\n[ARGUS] Error: {e}\n"

    def reset(self):
        self._memory = []

    def list_models(self) -> list:
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []

    def _model_err(self) -> str:
        models = self.list_models()
        hint   = ", ".join(models) if models else "run: ollama list"
        return f"\n[ARGUS] Model '{self.model}' not found. Available: {hint}\n"
