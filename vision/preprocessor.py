import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ProcessedImage:
    original: np.ndarray
    gray: np.ndarray
    denoised: np.ndarray
    edges: np.ndarray
    width: int
    height: int
    filepath: Optional[str] = None


PreprocessedImage = ProcessedImage


class ImagePreprocessor:
    TARGET_WIDTH = 1280

    def preprocess(self, source) -> ProcessedImage:
        filepath = None
        if isinstance(source, str):
            filepath = source
            img = cv2.imread(source)
            if img is None:
                raise ValueError(f"Cannot load image: {source}")
        else:
            img = source.copy()

        img    = self._resize(img)
        gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        den    = cv2.fastNlMeansDenoising(gray, h=10)
        sharp  = cv2.filter2D(den, -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))
        edges  = cv2.Canny(sharp, 50, 150)

        return ProcessedImage(
            original=img, gray=gray, denoised=den, edges=edges,
            width=img.shape[1], height=img.shape[0], filepath=filepath
        )

    def _resize(self, img):
        h, w = img.shape[:2]
        if w == self.TARGET_WIDTH:
            return img
        scale = self.TARGET_WIDTH / w
        return cv2.resize(img, (self.TARGET_WIDTH, int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
