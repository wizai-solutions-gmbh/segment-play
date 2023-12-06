from typing import Optional

import cv2
import numpy as np

COLORS = [
    (47, 79, 79),
    (165, 42, 42),
    (46, 139, 87),
    (128, 128, 0),
    (0, 0, 139),
    (255, 0, 0),
    (255, 165, 0),
    (255, 255, 0),
    (124, 252, 0),
    (186, 85, 211),
    (0, 250, 154),
    (0, 255, 255),
    (0, 0, 255),
    (255, 0, 255),
    (30, 144, 255),
    (240, 230, 140),
    (221, 160, 221),
    (255, 20, 147),
    (255, 160, 122),
    (135, 206, 250),
]


def show_box(
    image: np.ndarray,
    box: np.ndarray,
    id: Optional[int] = None
) -> np.ndarray:
    x0, y0 = int(box[0]), int(box[1])
    x1, y1 = int(box[2]), int(box[3])
    color = COLORS[2]
    if id is not None:
        color = COLORS[id % 20]
    return cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
