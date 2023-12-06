from typing import Tuple

import pytest

from frame.camera import get_codec


@pytest.mark.parametrize('codec_match', [(1196444237.0, 'MJPG')])
def test_get_codec(codec_match: Tuple[float, str]) -> None:
    codec_float, codec_name = codec_match
    assert get_codec(codec_float) == codec_name
