from typing import Optional, Tuple

import cv2
import numpy as np

from util.image import create_black_image


class Background:
    def __init__(self) -> None:
        self.avg: Optional[np.ndarray] = None

    def add_black(self, shape: Tuple[int, ...]) -> None:
        black_image = create_black_image(shape)
        if self.avg is None:
            self.avg = black_image.astype(np.float32)
        else:
            cv2.accumulateWeighted(black_image, self.avg, 0.05)

    def add_frame(self, img: np.ndarray) -> None:
        if self.avg is None:
            self.avg = img.astype(np.float32)
        else:
            cv2.accumulateWeighted(img, self.avg, 0.05)

    def get_bg(self) -> np.ndarray:
        assert self.avg is not None
        return cv2.convertScaleAbs(self.avg)
