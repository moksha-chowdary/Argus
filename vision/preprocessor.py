"""
ARGUS Vision Layer — Image Preprocessor
Handles chart screenshot ingestion and normalization.
"""
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class PreprocessedImage:
    original: np.ndarray
    gray: np.ndarray
    denoised: np.ndarray
    edges: np.ndarray
    width: int
    height: int
    filepath: Optional[str] = None


class ImagePreprocessor:
    """
    Prepares chart screenshots for downstream CV analysis and OCR.
    Pipeline: load → resize → denoise → sharpen → edge detect
    """

    TARGET_WIDTH = 1280  # normalize to consistent width

    def preprocess(self, source: str | np.ndarray) -> PreprocessedImage:
        """
        Accept a file path or an already-loaded numpy array.
        Returns a PreprocessedImage with all intermediate stages.
        """
        filepath = None
        if isinstance(source, (str, Path)):
            filepath = str(source)
            img = cv2.imread(filepath)
            if img is None:
                raise ValueError(f"Could not load image from {filepath}")
        else:
            img = source.copy()

        img = self._normalize_size(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = self._denoise(gray)
        sharpened = self._sharpen(denoised)
        edges = self._detect_edges(sharpened)

        return PreprocessedImage(
            original=img,
            gray=gray,
            denoised=denoised,
            edges=edges,
            width=img.shape[1],
            height=img.shape[0],
            filepath=filepath,
        )

    def _normalize_size(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if w == self.TARGET_WIDTH:
            return img
        scale = self.TARGET_WIDTH / w
        new_h = int(h * scale)
        return cv2.resize(img, (self.TARGET_WIDTH, new_h), interpolation=cv2.INTER_LANCZOS4)

    def _denoise(self, gray: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    def _sharpen(self, gray: np.ndarray) -> np.ndarray:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(gray, -1, kernel)

    def _detect_edges(self, gray: np.ndarray) -> np.ndarray:
        return cv2.Canny(gray, threshold1=50, threshold2=150)

    def save_debug_stages(self, processed: PreprocessedImage, out_dir: str = "/tmp/argus_debug"):
        """Dump intermediate stages for debugging the vision pipeline."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "original.png"), processed.original)
        cv2.imwrite(str(out / "gray.png"), processed.gray)
        cv2.imwrite(str(out / "denoised.png"), processed.denoised)
        cv2.imwrite(str(out / "edges.png"), processed.edges)
