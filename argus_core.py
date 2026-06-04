"""
ARGUS Core Orchestrator
Ties Vision → Intelligence → Conversation into a single pipeline call.
"""
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Generator
import numpy as np

from vision.preprocessor import ImagePreprocessor
from vision.ocr_pipeline import ChartOCRPipeline
from vision.cv_analyzer import CVAnalyzer
from intelligence.market_summary import MarketSummaryBuilder, MarketSummary
from intelligence.signal_generator import SignalGenerator, TradingSignal
from conversation.llm_client import ReasoningPipeline, OllamaClient


@dataclass
class ArgusResult:
    summary: MarketSummary
    signal: TradingSignal
    llm_response: str
    processing_time_ms: float


class ArgusOrchestrator:
    """
    Single entry point for the full ARGUS pipeline.
    
    Usage:
        argus = ArgusOrchestrator()
        result = argus.analyze("chart.png", "Is this a good breakout setup?")
        print(result.llm_response)
    """

    def __init__(
        self,
        ollama_model: str = "gemma3",
        ollama_url: str = "http://localhost:11434/api",
    ):
        self.preprocessor = ImagePreprocessor()
        self.ocr = ChartOCRPipeline()
        self.cv = CVAnalyzer()
        self.summary_builder = MarketSummaryBuilder()
        self.signal_gen = SignalGenerator()
        self.llm_client = OllamaClient(base_url=ollama_url, model=ollama_model)
        self.reasoning = ReasoningPipeline(client=self.llm_client)

    # ─── Main pipeline ────────────────────────────────────────────────────────

    def analyze(
        self,
        image_source: str | np.ndarray,
        user_question: str = "Analyze this chart and give me your assessment.",
        stream: bool = False,
    ) -> ArgusResult | Generator[str, None, None]:
        """
        Full pipeline: image → vision → intelligence → LLM response.
        
        Args:
            image_source: File path or numpy array (BGR)
            user_question: User's chat message
            stream: If True, returns a generator yielding tokens
        
        Returns:
            ArgusResult with full analysis, or a token generator if stream=True
        """
        t0 = time.time()

        # 1. Vision layer
        processed = self.preprocessor.preprocess(image_source)
        ocr_result = self.ocr.extract(processed)
        cv_result = self.cv.analyze(processed)

        # 2. Intelligence layer
        summary = self.summary_builder.build(cv_result, ocr_result)
        signal = self.signal_gen.generate(summary)

        # 3. Conversation layer
        if stream:
            return self.reasoning.analyze_chart(summary, signal, user_question, stream=True)

        llm_response = self.reasoning.analyze_chart(summary, signal, user_question, stream=False)
        elapsed = (time.time() - t0) * 1000

        return ArgusResult(
            summary=summary,
            signal=signal,
            llm_response=llm_response,
            processing_time_ms=round(elapsed, 1),
        )

    def chat(
        self,
        message: str,
        stream: bool = True,
    ) -> str | Generator[str, None, None]:
        """
        Follow-up conversation after a chart has been analyzed.
        Maintains in-session memory.
        """
        return self.reasoning.follow_up(message, stream=stream)

    def reset(self):
        """Clear session state for a new chart upload."""
        self.reasoning.reset_session()

    # ─── Vision-only mode (for debugging / testing) ───────────────────────────

    def extract_only(
        self, image_source: str | np.ndarray
    ) -> tuple[MarketSummary, TradingSignal]:
        """Run Vision + Intelligence layers without calling the LLM."""
        processed = self.preprocessor.preprocess(image_source)
        ocr_result = self.ocr.extract(processed)
        cv_result = self.cv.analyze(processed)
        summary = self.summary_builder.build(cv_result, ocr_result)
        signal = self.signal_gen.generate(summary)
        return summary, signal

    # ─── Model info ───────────────────────────────────────────────────────────

    def list_available_models(self) -> list[str]:
        return self.llm_client.list_models()

    def switch_model(self, model_name: str):
        self.llm_client.model = model_name
