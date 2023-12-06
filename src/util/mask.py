from typing import Optional, Tuple

import cv2
import numpy as np
from skimage.transform import resize

from util.image import clip_section


def create_empty_mask(shape: Tuple[int, ...]) -> np.ndarray:
    return np.zeros(shape, dtype=bool)


def add_masks(
        mask: np.ndarray,
        mask2: np.ndarray,
        position: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    if position is None:
        mask = np.logical_or(mask, mask2)
    else:
        h = mask2.shape[0]
        w = mask2.shape[1]
        x = position[1]
        y = position[0]
        mask[y:y + h, x:x + w] = np.logical_or(mask[y:y + h, x:x + w], mask2)
    return mask


def erode(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((5, 5), dtype=np.float32)
    mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=3)
    return mask.astype(bool)


def dilate(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((5, 5), dtype=np.float32)
    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=3)
    return mask.astype(bool)


def scale_mask(mask: np.ndarray, scale: float) -> np.ndarray:
    width = int(mask.shape[1] * scale)
    height = int(mask.shape[0] * scale)
    dim = (height, width)
    scaled_mask = resize(mask, dim, order=0)
    return scaled_mask


def apply_mask(
    base: np.ndarray,
    image: np.ndarray,
    mask: np.ndarray,
    position: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    masked_image = base
    if position is None:
        masked_image[:, :, 0] = base[:, :, 0] * \
            (1.0 - mask) + image[:, :, 0] * mask
        masked_image[:, :, 1] = base[:, :, 1] * \
            (1.0 - mask) + image[:, :, 1] * mask
        masked_image[:, :, 2] = base[:, :, 2] * \
            (1.0 - mask) + image[:, :, 2] * mask
    else:
        h = mask.shape[0]
        w = mask.shape[1]
        x = position[1]
        y = position[0]
        x, y, w, h = clip_section(x, y, w, h, base)
        masked_image[y:y + h, x:x + w, 0] = base[y:y + h, x:x + w, 0] * \
            (1.0 - mask[:h, :w]) + image[y:y + h, x:x + w, 0] * mask[:h, :w]
        masked_image[y:y + h, x:x + w, 1] = base[y:y + h, x:x + w, 1] * \
            (1.0 - mask[:h, :w]) + image[y:y + h, x:x + w, 1] * mask[:h, :w]
        masked_image[y:y + h, x:x + w, 2] = base[y:y + h, x:x + w, 2] * \
            (1.0 - mask[:h, :w]) + image[y:y + h, x:x + w, 2] * mask[:h, :w]
    return masked_image


def apply_mask_grayscale(
    base: np.ndarray,
    image: np.ndarray,
    mask: np.ndarray,
    gray: bool = False,
    position: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    if gray:
        masked_image = base
        if position is None:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            masked_image[:, :, 0] = base[:, :, 0] * \
                (1.0 - mask) + gray_image * mask
            masked_image[:, :, 1] = base[:, :, 1] * \
                (1.0 - mask) + gray_image * mask
            masked_image[:, :, 2] = base[:, :, 2] * \
                (1.0 - mask) + gray_image * mask
        else:
            h = mask.shape[0]
            w = mask.shape[1]
            x = position[1]
            y = position[0]
            x, y, w, h = clip_section(x, y, w, h, base)
            gray_image = image[y:y + h, x:x + w]
            gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
            masked_image[y:y + h, x:x + w, 0] = base[y:y + h, x:x + w, 0] * \
                (1.0 - mask[:h, :w]) + gray_image * mask[:h, :w]
            masked_image[y:y + h, x:x + w, 1] = base[y:y + h, x:x + w, 1] * \
                (1.0 - mask[:h, :w]) + gray_image * mask[:h, :w]
            masked_image[y:y + h, x:x + w, 2] = base[y:y + h, x:x + w, 2] * \
                (1.0 - mask[:h, :w]) + gray_image * mask[:h, :w]

        return masked_image
    else:
        return apply_mask(base, image, mask, position)
