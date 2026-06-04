import re
import sys
import cv2
import numpy as np
import pytesseract
from dataclasses import dataclass, field
from typing import Optional

# Windows: point to Tesseract binary
if sys.platform == "win32":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

_CFG = "--oem 3 --psm 6"
_PRICE  = re.compile(r"\b(\d{1,7}(?:[.,]\d{1,4})?)\b")
_RSI    = re.compile(r"RSI[^0-9]*([\d.]+)", re.I)
_MACD   = re.compile(r"MACD[^-\d]*([-\d.]+)", re.I)
_TF     = re.compile(r"\b(1m|3m|5m|15m|30m|1h|2h|4h|1d|1w|1M)\b")
_TICKER = re.compile(r"\b([A-Z]{2,5}(?:USDT?|USD|INR|BTC|ETH)?)\b")


@dataclass
class OCRResult:
    raw_text: str
    price_levels: list
    rsi_value: Optional[float]
    macd_value: Optional[float]
    timeframe: Optional[str]
    ticker: Optional[str]
    extra: dict = field(default_factory=dict)


class ChartOCR:
    def extract(self, processed) -> OCRResult:
        h, w = processed.gray.shape
        regions = {
            "title":     processed.gray[0:int(h*.07), :],
            "price_axis":processed.gray[int(h*.07):int(h*.65), int(w*.90):],
            "indicators":processed.gray[int(h*.65):int(h*.88), :],
        }
        texts = {k: self._ocr(v) for k, v in regions.items()}
        full  = "\n".join(texts.values())

        return OCRResult(
            raw_text    = full,
            price_levels= self._prices(texts["price_axis"]),
            rsi_value   = self._float(full, _RSI),
            macd_value  = self._float(full, _MACD),
            timeframe   = self._match(full, _TF),
            ticker      = self._match(texts["title"] + texts["price_axis"], _TICKER),
            extra       = {"region_texts": texts},
        )

    def _ocr(self, region):
        if region.size == 0:
            return ""
        scale = max(1, 50 // max(region.shape[0], 1))
        if scale > 1:
            region = cv2.resize(region, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        bw = cv2.adaptiveThreshold(region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 15, 4)
        return pytesseract.image_to_string(bw, config=_CFG).strip()

    def _prices(self, text):
        out = []
        for m in _PRICE.findall(text):
            try:
                out.append(float(m.replace(",", "")))
            except ValueError:
                pass
        return sorted(set(out))

    def _float(self, text, pat):
        m = pat.search(text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        return None

    def _match(self, text, pat):
        m = pat.search(text)
        return m.group(1) if m else None
