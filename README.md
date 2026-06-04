# ARGUS V4 — NSE Intelligence Terminal

## Setup (one time)

```bash
pip install -r requirements.txt
# Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
```

## Run

```bash
# Normal start (live feed ON by default)
python argus.py

# Start with a chart immediately
python argus.py chart.png

# Start with live feed OFF (no internet needed)
python argus.py --no-live

# Start in training mode
python argus.py --train
```

## Key Commands

| Command | What it does |
|---|---|
| `load chart.png` | Analyze a chart image |
| `livefeed on` | Enable live prices + news |
| `livefeed off` | Disable — analysis uses chart + memory only |
| `prices` | Show live NSE watchlist |
| `news` | Today's sector sentiment |
| `outcome abc123 win 1320` | Record trade outcome + update memory |
| `pending` | Trades awaiting outcome |
| `stats` | Performance dashboard |
| `history ICICIBANK` | Past trades for a stock |
| `train` | Training mode — chart → predict → outcome → self-adjust |
| `reload` | Reload rulebook.json without restarting |

## Training Workflow

1. `python argus.py --train`
2. Give it a past chart → ARGUS predicts
3. Tell it `win` or `loss`
4. ARGUS writes updated reliability into `data/rulebook.json`
5. After 20+ trades the memory score starts influencing signals

## Live Feed Toggle

The toggle persists across sessions in `data/state.json`.

- **ON** → 3-layer signal: chart + news sentiment + memory
- **OFF** → 2-layer signal: chart + memory only (good for backtesting or no-internet sessions)

## Editing Rules

Open `data/rulebook.json` in any editor. Change weights, add notes, edit reliability scores.
Then run `reload` inside ARGUS — no restart needed.

## Resume Note

This project demonstrates:
- Local LLM deployment (Gemma4 on RTX 4050)
- Computer vision for financial chart analysis
- Persistent vector memory (ChromaDB)
- Configurable ML scoring system (JSON rulebook)
- Real-time NSE data integration (yfinance)
- Feedback-driven learning loop
