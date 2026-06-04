"""ARGUS V4 — Vision Layer"""

import cv2, re, sys
import numpy as np
from dataclasses import dataclass
from typing import Optional

try:
    import pytesseract
    from config import TESSERACT_CMD
    if sys.platform == "win32":
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False


@dataclass
class MarketSummary:
    ticker:     Optional[str]   = None
    timeframe:  Optional[str]   = None
    trend:      str             = "unknown"
    strength:   str             = "unknown"
    support:    Optional[float] = None
    resistance: Optional[float] = None
    rsi:        Optional[float] = None
    rsi_state:  str             = "neutral"
    volume:     str             = "normal"
    pattern:    str             = "none"
    momentum:   str             = "neutral"
    volatility: str             = "normal"
    raw_text:   str             = ""


class ChartAnalyzer:
    def analyze(self, img_path: str) -> Optional[MarketSummary]:
        img = cv2.imread(img_path)
        if img is None:
            return None
        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w    = gray.shape
        summary = MarketSummary()
        summary.raw_text = self._ocr(img)
        self._extract_meta(summary)
        self._analyze_trend(gray, summary, h, w)
        self._analyze_volume(gray, h, summary)
        self._detect_patterns(gray, summary, h, w)
        self._classify_rsi(summary)
        return summary

    def _ocr(self, img) -> str:
        if not OCR_AVAILABLE:
            return ""
        try:
            gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return pytesseract.image_to_string(thr)
        except Exception:
            return ""

    def _extract_meta(self, s: MarketSummary):
        txt = s.raw_text.upper()
        m   = re.search(r'\b([A-Z]{3,10}(?:BANK|FIN|LTD)?)\b', txt)
        if m:
            s.ticker = m.group(1)
        for tf in ["1D", "4H", "1H", "30M", "15M", "5M"]:
            if tf in txt:
                s.timeframe = tf.lower()
                break
        m = re.search(r'RSI[:\s]*([\d.]+)', txt)
        if m:
            try: s.rsi = float(m.group(1))
            except Exception: pass

    def _analyze_trend(self, gray, s: MarketSummary, h, w):
        mid   = w // 2
        diff  = float(np.mean(gray[:, mid:])) - float(np.mean(gray[:, :mid]))
        if diff > 8:
            s.trend    = "bullish"
            s.strength = "strong" if diff > 15 else "moderate"
            s.momentum = "positive"
        elif diff < -8:
            s.trend    = "bearish"
            s.strength = "strong" if diff < -15 else "moderate"
            s.momentum = "negative"
        else:
            s.trend    = "sideways"
            s.strength = "weak"
            s.momentum = "neutral"
        edges  = cv2.Canny(gray, 50, 150)
        h_proj = np.sum(edges, axis=1).astype(float)
        peaks  = np.argsort(h_proj)[-5:]
        ys     = sorted(peaks)
        if len(ys) >= 2:
            s.resistance = round(1000 + (h - ys[0])  / h * 500, 0)
            s.support    = round(1000 + (h - ys[-1]) / h * 500, 0)

    def _analyze_volume(self, gray, h, s: MarketSummary):
        br = float(np.mean(gray[int(h*0.8):, :]))
        s.volume = "high" if br > 160 else ("low" if br < 80 else "normal")

    def _detect_patterns(self, gray, s: MarketSummary, h, w):
        right   = gray[:, w//2:]
        col_max = np.max(right, axis=0).astype(float)
        diffs   = np.diff(col_max)
        neg     = np.sum(diffs < -1)
        pos     = np.sum(diffs > 1)
        if neg > pos * 1.5:
            s.pattern = "bearish channel / lower highs"
        elif pos > neg * 1.5:
            s.pattern = "bullish trend / higher highs"
        else:
            s.pattern = "sideways / consolidation"
        std = float(np.std(col_max[-20:]))
        if std < 5:
            s.pattern   += " + volatility contraction"
            s.volatility = "low"
        elif std > 20:
            s.volatility = "high"

    def _classify_rsi(self, s: MarketSummary):
        if s.rsi is None:
            return
        if s.rsi >= 70: s.rsi_state = "overbought"
        elif s.rsi <= 30: s.rsi_state = "oversold"
        else: s.rsi_state = "neutral"
