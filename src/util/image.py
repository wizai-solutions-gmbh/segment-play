from typing import Tuple

import cv2
import numpy as np


def create_black_image(shape: Tuple[int, ...]) -> np.ndarray:
    return np.zeros(shape, dtype=np.uint8)


def scale_image(image: np.ndarray, scale: float) -> np.ndarray:
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def clip_section(
    x: int,
    y: int,
    w: int,
    h: int,
    image: np.ndarray
) -> Tuple[int, int, int, int]:
    if x < 0:
        w = w + x
        x = 0
    if y < 0:
        h = h + y
        y = 0
    if x + w > image.shape[1]:
        w = image.shape[1] - x
    if y + h > image.shape[0]:
        h = image.shape[0] - y
    return x, y, w, h


def clip_section_xyxy(
    x: int,
    y: int,
    x2: int,
    y2: int,
    image: np.ndarray
) -> Tuple[int, int, int, int]:
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x2 > image.shape[1]:
        x2 = image.shape[1]
    if y2 > image.shape[0]:
        y2 = image.shape[0]
    return x, y, x2, y2
