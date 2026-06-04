import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CVResult:
    trend: str              # bullish | bearish | sideways
    strength: str           # strong | moderate | weak
    support_norm: list      # normalized 0-1 y positions
    resistance_norm: list
    candles: list
    volume: str             # increasing | decreasing | flat
    volatility: str         # high | moderate | low | compressed
    pattern: Optional[str]
    consolidation: bool


class CVAnalyzer:
    def analyze(self, processed) -> CVResult:
        h, w = processed.gray.shape
        chart  = processed.gray[int(h*.07):int(h*.65), :int(w*.92)]
        volume = processed.gray[int(h*.75):,           :int(w*.92)]

        candles   = self._candles(chart)
        trend, st = self._trend(chart, candles)
        sup, res  = self._levels(chart)
        vol       = self._volume(volume)
        vola      = self._volatility(chart)
        consol    = self._consolidation(chart)
        pattern   = self._pattern(chart, trend)

        return CVResult(
            trend=trend, strength=st,
            support_norm=sup, resistance_norm=res,
            candles=candles, volume=vol,
            volatility=vola, pattern=pattern,
            consolidation=consol,
        )

    def _candles(self, chart):
        _, th = cv2.threshold(chart, 160, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = chart.shape
        out = []
        for c in cnts:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw == 0:
                continue
            ar = bh / bw
            if 1.5 < ar < 12 and bw < w*.03 and 3 < bh < h*.5:
                mean = chart[y:y+bh, x:x+bw].mean()
                out.append({"x": x, "y": y, "w": bw, "h": bh,
                            "type": "bear" if mean < 127 else "bull"})
        return sorted(out, key=lambda c: c["x"])

    def _trend(self, chart, candles):
        if len(candles) >= 5:
            xs = np.array([c["x"] + c["w"]/2 for c in candles], dtype=float)
            ys = np.array([c["y"] + c["h"]/2 for c in candles], dtype=float)
            slope, _ = np.polyfit(xs, ys, 1)
            n = slope / chart.shape[0]
            if abs(n) < 0.0005:
                return "sideways", "moderate"
            if n < 0:
                return "bullish", "strong" if n < -0.001 else "moderate"
            return "bearish", "strong" if n > 0.001 else "moderate"
        return "sideways", "weak"

    def _levels(self, chart):
        edges = cv2.Canny(chart, 30, 100)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                                minLineLength=int(chart.shape[1]*.2), maxLineGap=20)
        ys = []
        if lines is not None:
            for l in lines:
                x1,y1,x2,y2 = l[0]
                if abs(y2-y1) < 5:
                    ys.append(round(((y1+y2)/2) / chart.shape[0], 3))
        ys = self._cluster(ys)
        sup = sorted([y for y in ys if y >= 0.5])[:3]
        res = sorted([y for y in ys if y < 0.5])[:3]
        return sup, res

    def _cluster(self, levels, tol=0.03):
        if not levels:
            return []
        levels = sorted(levels)
        groups = [[levels[0]]]
        for v in levels[1:]:
            if v - groups[-1][-1] < tol:
                groups[-1].append(v)
            else:
                groups.append([v])
        return [round(sum(g)/len(g), 3) for g in groups]

    def _volume(self, region):
        if region.size == 0:
            return "unknown"
        w = region.shape[1]
        t = w // 3
        diff = region[:, 2*t:].mean() - region[:, :t].mean()
        if abs(diff) < 5:
            return "flat"
        return "increasing" if diff > 0 else "decreasing"

    def _volatility(self, chart):
        s = np.std(chart, axis=0).mean()
        if s < 20: return "compressed"
        if s < 35: return "low"
        if s < 55: return "moderate"
        return "high"

    def _consolidation(self, chart):
        recent = chart[:, int(chart.shape[1]*.7):]
        return float(np.std(recent.max(axis=1) - recent.min(axis=1))) < 15.0

    def _pattern(self, chart, trend):
        edges = cv2.Canny(chart, 40, 120)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 60,
                                minLineLength=int(chart.shape[1]*.15), maxLineGap=15)
        if lines is None:
            return None
        slopes = []
        for l in lines:
            x1,y1,x2,y2 = l[0]
            if x2 != x1:
                slopes.append((y2-y1)/(x2-x1))
        if not slopes:
            return None
        pos = sum(1 for s in slopes if s >  0.05)
        neg = sum(1 for s in slopes if s < -0.05)
        flat= sum(1 for s in slopes if abs(s) <= 0.05)
        if trend == "bullish" and flat > pos and flat > neg: return "ascending_triangle"
        if trend == "bearish" and flat > pos and flat > neg: return "descending_triangle"
        if pos > 2 and neg > 2:                              return "symmetrical_triangle"
        if trend == "bullish" and neg > pos:                 return "bull_flag"
        if trend == "bearish" and pos > neg:                 return "bear_flag"
        return None
