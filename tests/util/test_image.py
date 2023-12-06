from typing import Tuple

import numpy as np
import pytest

from util.image import clip_section, create_black_image, scale_image


@pytest.mark.parametrize('shape', [(20, 10), (10, 10, 3)])
def test_black(shape: Tuple[int, ...]) -> None:
    black_image = create_black_image(shape)
    assert black_image.shape == shape
    assert np.sum(black_image) == 0


@pytest.mark.parametrize('scale', [1.0, 0.5, 2.0])
def test_scale_image(scale: float) -> None:
    black_image = create_black_image((10, 10, 3))
    scaled_image = scale_image(black_image, scale)
    assert scaled_image.shape == (10 * scale, 10 * scale, 3)
    assert np.sum(scaled_image) == 0


@pytest.mark.parametrize('data', [
    ((-5, -5, 110, 110), (100, 100), (0, 0, 100, 100)),
    ((-5, -5, 95, 95), (100, 100), (0, 0, 90, 90)),
    ((15, 15, 105, 105), (100, 100), (15, 15, 85, 85)),
    ((-5, 15, 110, 70), (100, 100), (0, 15, 100, 70)),
    ((15, 15, 105, 105), (200, 200), (15, 15, 105, 105))
])
def test_clip_section(
    data: Tuple[Tuple[int, int, int, int],
                Tuple[int, int], Tuple[int, int, int, int]]
) -> None:
    box, image_size, expected_box = data
    black_image = create_black_image((image_size[1], image_size[0], 3))
    assert clip_section(*box, black_image) == expected_box
