"""
ARGUS V5 — FinBERT Financial Sentiment Engine
Scores financial headlines using ProsusAI/finbert (HuggingFace transformers)
with calibrated softmax probabilities: Compound = P(positive) - P(negative).
Includes a resilient local lexicon fallback for zero-latency or offline execution.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Suppress transformer warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

FINBERT_MODEL_NAME = "ProsusAI/finbert"

# Financial domain lexicon for fallback / fast offline scoring
BULLISH_FIN_LEXICON = {
    "surge": 0.8, "rally": 0.7, "gain": 0.5, "jump": 0.6, "strong": 0.5,
    "profit": 0.6, "beat": 0.8, "record": 0.7, "upgrade": 0.8, "buy": 0.7,
    "outperform": 0.8, "bullish": 0.8, "expansion": 0.6, "contract": 0.5,
    "growth": 0.6, "dividend": 0.5, "recovery": 0.6, "boost": 0.6,
    "inflow": 0.7, "fii buying": 0.9, "dii buying": 0.8, "order win": 0.8,
    "acquisition": 0.6, "breakout": 0.7, "all-time high": 0.9, "ebitda up": 0.8,
}

BEARISH_FIN_LEXICON = {
    "fall": -0.6, "drop": -0.6, "decline": -0.5, "loss": -0.7, "weak": -0.6,
    "miss": -0.8, "downgrade": -0.8, "sell": -0.7, "underperform": -0.8,
    "risk": -0.5, "concern": -0.5, "pressure": -0.5, "bearish": -0.8,
    "cut": -0.6, "slowdown": -0.6, "warn": -0.7, "crash": -0.9, "plunge": -0.9,
    "outflow": -0.7, "fii selling": -0.9, "penalty": -0.8, "fraud": -1.0,
    "investigation": -0.7, "default": -1.0, "debt": -0.6, "npa surge": -0.9,
}


class FinBERTSentimentAnalyzer:
    """
    FinBERT NLP sentiment pipeline for financial news text.
    Produces calibrated continuous scores in [-1.0, 1.0].
    """

    def __init__(self, use_gpu: bool = False, model_name: str = FINBERT_MODEL_NAME):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._is_loaded = False
        self._load_attempted = False

    def _lazy_load_finbert(self) -> bool:
        if self._is_loaded:
            return True
        if self._load_attempted:
            return False

        self._load_attempted = True
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            self._torch = torch
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.eval()

            if self.use_gpu and torch.cuda.is_available():
                self._model = self._model.cuda()

            self._is_loaded = True
            return True
        except Exception as e:
            # Fallback to domain lexicon smoothly
            self._is_loaded = False
            return False

    def score_text(self, text: str) -> Dict[str, float]:
        """
        Calculates sentiment probability distribution and compound score for a single string.
        Returns:
            {
                'positive': float,
                'negative': float,
                'neutral': float,
                'compound': float,  # P(pos) - P(neg) in [-1.0, 1.0]
                'method': 'finbert' | 'lexicon'
            }
        """
        if not text or not text.strip():
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "compound": 0.0, "method": "none"}

        # Try FinBERT if model is available
        if self._lazy_load_finbert():
            try:
                inputs = self._tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=128, padding=True
                )
                if self.use_gpu and self._torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                with self._torch.no_grad():
                    outputs = self._model(**inputs)
                    probs = self._torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

                # ProsusAI/finbert output class order: [positive, negative, neutral]
                p_pos = float(probs[0])
                p_neg = float(probs[1])
                p_neu = float(probs[2])
                compound = p_pos - p_neg

                return {
                    "positive": round(p_pos, 4),
                    "negative": round(p_neg, 4),
                    "neutral": round(p_neu, 4),
                    "compound": round(max(-1.0, min(1.0, compound)), 4),
                    "method": "finbert",
                }
            except Exception:
                pass

        # Robust Fallback to Indian Financial Lexicon
        return self._score_lexicon(text)

    def _score_lexicon(self, text: str) -> Dict[str, float]:
        clean = text.lower()
        score = 0.0
        pos_hits = 0
        neg_hits = 0

        for word, weight in BULLISH_FIN_LEXICON.items():
            if word in clean:
                score += weight
                pos_hits += 1

        for word, weight in BEARISH_FIN_LEXICON.items():
            if word in clean:
                score += weight  # weight is negative
                neg_hits += 1

        total_hits = pos_hits + neg_hits
        if total_hits == 0:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "compound": 0.0, "method": "lexicon"}

        compound = max(-1.0, min(1.0, score / max(1.0, total_hits)))
        p_pos = max(0.0, compound) if compound > 0 else 0.0
        p_neg = max(0.0, -compound) if compound < 0 else 0.0
        p_neu = 1.0 - (p_pos + p_neg)

        return {
            "positive": round(p_pos, 4),
            "negative": round(p_neg, 4),
            "neutral": round(max(0.0, p_neu), 4),
            "compound": round(compound, 4),
            "method": "lexicon",
        }

    def score_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Batch score a list of headlines."""
        return [self.score_text(t) for t in texts]
