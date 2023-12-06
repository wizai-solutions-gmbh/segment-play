
from typing import Tuple

import numpy as np
import pytest

from util.image import create_black_image
from util.mask import (add_masks, apply_mask_grayscale, create_empty_mask,
                       dilate, erode, scale_mask)


@pytest.mark.parametrize('first_shape',
                         [(20, 20), (10, 10), (10, 10, 3)])
@pytest.mark.parametrize('second_shape',
                         [(20, 20), (10, 10), (10, 10, 3)])
@pytest.mark.parametrize('position', [None, (5, 5), (10, 10)])
def test_add_masks(
    first_shape: Tuple[int, ...],
    second_shape: Tuple[int, ...],
    position: Tuple[int, int]
) -> None:
    first_mask = np.empty(first_shape, dtype=bool)
    second_mask = np.ones(second_shape, dtype=bool)

    mismatch_size = False
    if len(first_shape) != len(second_shape):
        mismatch_size = True
    else:
        if position:
            required_h = second_shape[0] + position[0]
            required_w = second_shape[1] + position[1]
            if first_shape[0] < required_h or \
                    first_shape[0] < required_w:
                mismatch_size = True
        else:
            for size_first, size_second in zip(first_shape, second_shape):
                if size_first != size_second:
                    mismatch_size = True

    if mismatch_size:
        with pytest.raises(Exception):
            combined_mask = add_masks(first_mask, second_mask, position)
    else:
        combined_mask = add_masks(first_mask, second_mask, position)
        assert combined_mask.shape == first_shape


def test_erode() -> None:
    blank_mask = create_empty_mask((100, 100))
    blank_mask[20:80, 20:80] = 1
    assert np.sum(blank_mask) == 60 * 60
    scaled_mask = erode(blank_mask)
    assert np.sum(scaled_mask) < 60 * 60


def test_dilate() -> None:
    blank_mask = create_empty_mask((100, 100))
    blank_mask[20:80, 20:80] = 1
    assert np.sum(blank_mask) == 60 * 60
    scaled_mask = dilate(blank_mask)
    assert np.sum(scaled_mask) > 60 * 60


@pytest.mark.parametrize('scale', [1.0, 0.5, 2.0])
def test_scale_mask(scale: float) -> None:
    blank_mask = create_empty_mask((10, 10))
    scaled_mask = scale_mask(blank_mask, scale)
    assert scaled_mask.shape == (10 * scale, 10 * scale)
    assert np.sum(scaled_mask) == 0


@pytest.mark.parametrize('mask_shape', [(10, 10), (20, 20)])
@pytest.mark.parametrize('position', [None, (0, 0), (10, 20)])
@pytest.mark.parametrize('gray', [True, False])
def test_apply_mask(
    mask_shape: Tuple[int, ...],
    position: Tuple[int, int],
    gray: bool
) -> None:
    width = 100
    height = 100
    value = 30
    gray_value = 9
    red_image = create_black_image((height, width, 3))
    red_image[:, :, 0] = value
    blue_image = create_black_image((height, width, 3))
    blue_image[:, :, 2] = value
    mask = np.ones(mask_shape)
    if position is None:
        full_mask = create_empty_mask((height, width))
        full_mask[:mask_shape[0], :mask_shape[1]] = mask[:, :]
        mask = full_mask

    merged_image = apply_mask_grayscale(
        red_image, blue_image, mask, gray, position)

    if position is None:
        position = (0, 0)

    mask_pixels = mask_shape[0] * mask_shape[1]

    assert merged_image.shape == red_image.shape
    if not gray:
        assert np.sum(merged_image[:, :, 0]) == (
            height * width - mask_pixels) * value
        assert np.sum(merged_image[:, :, 2]) == mask_pixels * value
        assert merged_image[position[0], position[1], 2] == value
    else:
        x = 0 if not position else position[1]
        y = 0 if not position else position[0]
        w = mask_shape[1]
        h = mask_shape[0]
        gray_r_sum = np.sum(merged_image[y:y + h, x:x + w, 0])
        assert np.sum(merged_image[:, :, 0]) - gray_r_sum \
            == (height * width - mask_pixels) * value
        assert np.sum(merged_image[:, :, 0]) == (
            width * height - mask_pixels) * value + gray_r_sum
        assert np.sum(merged_image[:, :, 1]) == mask_pixels * gray_value
        assert np.sum(merged_image[:, :, 2]) == mask_pixels * gray_value
        assert merged_image[position[0], position[1], 2] == gray_value
