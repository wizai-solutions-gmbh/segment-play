# flake8: noqa

import os.path
import sys
import time
from typing import Tuple

import numpy as np

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(sys.modules[__name__].__file__), '..', '..', 'src')))  # type: ignore  # noqa

from util.image import create_black_image
from util.mask import apply_mask

picture_size = (1440, 2560, 3)  # 4K resolution
repeat_count = 100


def full_mask_apply(size: Tuple[int, ...], repeat_count: int) -> None:
    red_image = create_black_image(size)
    red_image[:, :, 0] = 1
    blue_image = create_black_image(size)
    blue_image[:, :, 2] = 1
    mask = np.ones((size[0], size[1]))

    start_time = time.time()
    for _ in range(repeat_count):
        _ = apply_mask(red_image, blue_image, mask, None)
    end_time = time.time()
    print(end_time - start_time)


def part_mask_apply(size: Tuple[int, ...], repeat_count: int) -> None:
    red_image = create_black_image(size)
    red_image[:, :, 0] = 1
    blue_image = create_black_image(size)
    blue_image[:, :, 2] = 1
    mask = np.ones((300, 300))
    position = (100, 100)

    start_time = time.time()
    for _ in range(repeat_count):
        _ = apply_mask(red_image, blue_image, mask, position)
    end_time = time.time()
    print(end_time - start_time)


full_mask_apply(picture_size, repeat_count)
part_mask_apply(picture_size, repeat_count)
