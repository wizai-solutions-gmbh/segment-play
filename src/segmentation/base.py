from enum import Enum
from typing import Any, Optional

import numpy as np


class BodyPartSegmentation(Enum):
    ALL = 0
    LEFT_ARM = 1
    RIGHT_ARM = 2
    BOTH_ARMS = 3
    ONLY_FACE = 4


class Predictor:
    def set_image(self, image: np.ndarray) -> None:
        pass


class Segmentation:
    def __init__(self) -> None:
        self.predictor: Any = None
        self.image_embedding: Any = None

    def get_image_embedding(self) -> Any:
        return None

    def set_image(self, image: np.ndarray) -> None:
        self.predictor.set_image(image)
        self.image_embedding = self.get_image_embedding()

    def prepare_prompts(self, image: np.ndarray) -> None:
        pass

    def bbox_masks(
        self,
        bb: np.ndarray,
        points: Optional[np.ndarray] = None,
        point_modes: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return np.array([])
