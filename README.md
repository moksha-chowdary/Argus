# ARGUS — AI Visual Market Analyst

> Upload a chart screenshot. Get trading intelligence.

ARGUS is an offline conversational AI trading assistant that analyzes stock chart screenshots.
It is **not** an autonomous trading bot — it acts as an AI market analyst, reasoning about
visual chart data and communicating like a senior human trader.

---

## Architecture

```
chart.png
    │
    ▼
┌───────────────────────────────────┐
│         VISION LAYER              │
│  ImagePreprocessor → CVAnalyzer   │
│           └─ ChartOCRPipeline     │
└───────────────────────┬───────────┘
                        │ CVAnalysisResult + OCRResult
                        ▼
┌───────────────────────────────────┐
│       INTELLIGENCE LAYER          │
│  MarketSummaryBuilder             │
│           └─ SignalGenerator      │
└───────────────────────┬───────────┘
                        │ MarketSummary + TradingSignal
                        ▼
┌───────────────────────────────────┐
│       CONVERSATION LAYER          │
│  OllamaClient (Gemma 3)           │
│           └─ ReasoningPipeline    │
└───────────────────────┬───────────┘
                        │ streaming tokens
                        ▼
              Streamlit Chat UI
```

---

## Folder Structure

```
argus/
├── main.py                      # Entry point
├── argus_core.py                # Main orchestrator
├── requirements.txt
│
├── vision/
│   ├── preprocessor.py          # Image normalization
│   ├── ocr_pipeline.py          # Tesseract OCR extraction
│   └── cv_analyzer.py           # OpenCV pattern analysis
│
├── intelligence/
│   ├── market_summary.py        # Structured chart state
│   └── signal_generator.py      # BUY/SELL/WAIT signals
│
├── conversation/
│   └── llm_client.py            # Ollama client + prompting
│
├── api/
│   └── server.py                # FastAPI REST + WebSocket
│
├── ui/
│   └── app.py                   # Streamlit chat UI
│
├── tests/
│   └── test_pipeline.py         # Unit tests
│
└── docs/
    └── ROADMAP.md
```

---

## Setup

### 1. Install system dependencies

```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr tesseract-ocr-eng

# macOS
brew install tesseract
```

### 2. Install Python dependencies

```bash
cd argus
pip install -r requirements.txt
```

### 3. Install and start Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull Gemma 3 (choose based on your RAM)
ollama pull gemma3:4b     # ~3GB — works on 8GB RAM
ollama pull gemma3:12b    # ~8GB — better quality, needs 16GB RAM

# Start the Ollama server
ollama serve
```

### 4. Launch ARGUS

```bash
# Streamlit UI (recommended)
python main.py

# Or: FastAPI backend
python main.py api
```

Then open http://localhost:8501 in your browser.

---

## Usage

1. Upload a TradingView screenshot in the sidebar
2. Type a question in the chat: *"Is this a good breakout setup?"*
3. ARGUS runs the vision pipeline, generates a signal, and streams the LLM analysis
4. Ask follow-up questions about the same chart

### Example questions
- "Analyze this chart"
- "Where is the key support level?"
- "Is RSI showing divergence?"
- "What's the risk if I enter here?"
- "Should I wait for a retest?"

---

## Vision Pipeline — What It Detects

| Feature | Method |
|---------|--------|
| Trend direction + strength | Candle slope regression |
| Support / resistance zones | Horizontal Hough lines |
| Candlestick bodies | Contour detection |
| Volume behavior | Column intensity analysis |
| Volatility | Brightness std-dev |
| Chart pattern | Slope distribution heuristics |
| RSI value | Tesseract OCR |
| Price levels | Tesseract + regex |
| Ticker + timeframe | Tesseract + regex |

---

## V1 Limitations (be aware)

- Vision is heuristic, not CNN-based — works best on clean TradingView screenshots
- EMA line color detection is not yet implemented (inferred from trend)
- Support/resistance prices require OCR to succeed on the price axis
- Complex custom indicator panels may not parse correctly

---

## Roadmap → V2

- [ ] CNN-based candlestick classifier (replace heuristics)
- [ ] EMA color segmentation
- [ ] ChromaDB memory for chart history across sessions
- [ ] Multi-timeframe analysis (upload 3 charts at once)
- [ ] FAISS-based pattern similarity search
- [ ] Export analysis to PDF report
- [ ] Alert system: "notify me when this pattern forms"

---

## ⚠️ Disclaimer

ARGUS is an **educational tool** for learning technical analysis.
It is **not** financial advice. Never trade based solely on automated analysis.
Always do your own research and consult a qualified financial advisor.
