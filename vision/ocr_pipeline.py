"""
ARGUS Vision Layer — OCR Pipeline
Extracts price levels, indicator values, labels, and axis ticks
from chart screenshots using Tesseract.
"""
import re
import cv2
import numpy as np
import pytesseract
from dataclasses import dataclass, field
from typing import Optional
from vision.preprocessor import PreprocessedImage


@dataclass
class OCRResult:
    raw_text: str
    price_levels: list[float]
    rsi_value: Optional[float]
    macd_value: Optional[float]
    volume_labels: list[str]
    axis_labels: list[str]
    indicator_labels: list[str]
    timeframe: Optional[str]
    ticker: Optional[str]
    extra: dict = field(default_factory=dict)


class ChartOCRPipeline:
    """
    Extracts textual information from chart screenshots.

    Strategy:
      1. Crop regions of interest (price axis, indicator panels, title bar)
      2. Apply adaptive thresholding for binarization
      3. Run Tesseract with chart-optimized config
      4. Parse output with domain-specific regex
    """

    # Tesseract config — PSM 6 = assume single block, OEM 3 = best LSTM
    _TESS_CONFIG = "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,-+/%:()"

    # Regex patterns
    _PRICE_RE = re.compile(r"\b(\d{1,6}(?:[.,]\d{1,4})?)\b")
    _RSI_RE = re.compile(r"RSI[:\s(]*([\d.]+)", re.IGNORECASE)
    _MACD_RE = re.compile(r"MACD[:\s(]*([-\d.]+)", re.IGNORECASE)
    _TF_RE = re.compile(r"\b(1m|3m|5m|15m|30m|1h|2h|4h|1d|1D|1W|1M)\b")
    _TICKER_RE = re.compile(r"\b([A-Z]{2,5}(?:USDT?|USD|INR|BTC)?)\b")

    def extract(self, processed: PreprocessedImage) -> OCRResult:
        """Run full OCR extraction on a preprocessed chart image."""
        regions = self._extract_regions(processed)
        texts = {name: self._ocr_region(region) for name, region in regions.items()}
        full_text = "\n".join(texts.values())

        return OCRResult(
            raw_text=full_text,
            price_levels=self._extract_prices(texts.get("price_axis", "")),
            rsi_value=self._extract_indicator(full_text, self._RSI_RE),
            macd_value=self._extract_indicator(full_text, self._MACD_RE),
            volume_labels=self._extract_volume_labels(texts.get("volume_panel", "")),
            axis_labels=self._extract_generic_labels(texts.get("price_axis", "")),
            indicator_labels=self._extract_generic_labels(texts.get("indicator_panel", "")),
            timeframe=self._extract_timeframe(full_text),
            ticker=self._extract_ticker(texts.get("title_bar", "") + texts.get("price_axis", "")),
            extra={"region_texts": texts},
        )

    def _extract_regions(self, img: PreprocessedImage) -> dict[str, np.ndarray]:
        """
        Heuristic region cropping for typical TradingView layout.
        Adjust ratios if targeting other charting platforms.
        """
        h, w = img.gray.shape
        return {
            "title_bar":      img.gray[0:int(h * 0.06), :],
            "chart_main":     img.gray[int(h * 0.06):int(h * 0.65), :int(w * 0.92)],
            "price_axis":     img.gray[int(h * 0.06):int(h * 0.65), int(w * 0.90):],
            "indicator_panel":img.gray[int(h * 0.65):int(h * 0.85), :],
            "volume_panel":   img.gray[int(h * 0.85):, :],
        }

    def _ocr_region(self, region: np.ndarray) -> str:
        if region.size == 0:
            return ""
        # Upscale small regions for better accuracy
        scale = max(1, 60 // max(region.shape[0], 1))
        if scale > 1:
            region = cv2.resize(region, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        # Adaptive threshold for contrast normalization
        binarized = cv2.adaptiveThreshold(
            region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, blockSize=15, C=4
        )
        return pytesseract.image_to_string(binarized, config=self._TESS_CONFIG).strip()

    def _extract_prices(self, text: str) -> list[float]:
        matches = self._PRICE_RE.findall(text)
        prices = []
        for m in matches:
            try:
                prices.append(float(m.replace(",", "")))
            except ValueError:
                pass
        return sorted(set(prices))

    def _extract_indicator(self, text: str, pattern: re.Pattern) -> Optional[float]:
        m = pattern.search(text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        return None

    def _extract_volume_labels(self, text: str) -> list[str]:
        return [t for t in text.split() if t.replace(".", "").replace("K", "").replace("M", "").isalnum()]

    def _extract_generic_labels(self, text: str) -> list[str]:
        return [line.strip() for line in text.splitlines() if line.strip()]

    def _extract_timeframe(self, text: str) -> Optional[str]:
        m = self._TF_RE.search(text)
        return m.group(1).lower() if m else None

    def _extract_ticker(self, text: str) -> Optional[str]:
        m = self._TICKER_RE.search(text)
        return m.group(1) if m else None
