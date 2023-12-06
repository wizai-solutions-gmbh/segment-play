import os
import queue
import threading
from argparse import ArgumentParser
from dataclasses import dataclass
from queue import Queue as BasicQueue
from typing import Any, Dict, Optional

import cv2
import numpy as np


@dataclass
class CameraSettings:
    input: int = 0
    width: int = 1920
    height: int = 1080
    fps: int = 30
    codec: str = 'MJPG'
    api: Optional[int] = cv2.CAP_DSHOW if os.name == 'nt' else None


def add_camera_parameters(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('--cam-input', type=int,
                        default=0, help='Camera input to use')
    parser.add_argument('--cam-width', type=int,
                        default=1920, help='Camera image width')
    parser.add_argument('--cam-height', type=int,
                        default=1080, help='Camera image height')
    parser.add_argument('--cam-fps', type=int,
                        default=30, help='Camera fps')
    parser.add_argument('--cam-codec', type=str, default='MJPG',
                        help='Camera codec (e.g. MJPG, H264, YUV2)')

    return parser


def get_codec(cv2_codec_info: float) -> str:
    codec = int(cv2_codec_info)
    codec_str = str(chr(codec & 0xff) + chr((codec >> 8) & 0xff) +
                    chr((codec >> 16) & 0xff) + chr((codec >> 24) & 0xff))
    return codec_str


def parse_camera_settings(args: Dict[str, Any]) -> CameraSettings:
    return CameraSettings(
        args['cam_input'],
        args['cam_width'],
        args['cam_height'],
        args['cam_fps'],
        args['cam_codec'])


def check_camera(capture: cv2.VideoCapture, settings: CameraSettings) -> None:
    assert capture.get(cv2.CAP_PROP_FRAME_WIDTH) == settings.width, \
        'Camera setting width could not be set properly'
    assert capture.get(cv2.CAP_PROP_FRAME_HEIGHT) == settings.height, \
        'Camera setting height could not be set properly'
    codec = get_codec(capture.get(cv2.CAP_PROP_FOURCC))
    assert codec == settings.codec, \
        'Camera setting codec could not be set properly ' + codec
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    assert fps == settings.fps, \
        'Camera setting fps could not be set properly ' + str(fps)


def set_camera_parameters(
    capture: cv2.VideoCapture,
    settings: CameraSettings
) -> None:
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
    capture.set(cv2.CAP_PROP_FPS, settings.fps)
    capture.set(cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter.fourcc(*(settings.codec)))


class VideoCapture:
    def __init__(self, settings: Optional[CameraSettings] = None) -> None:
        if settings:
            self.cap = cv2.VideoCapture(settings.input, settings.api)
            set_camera_parameters(self.cap, settings)
            check_camera(self.cap, settings)
        else:
            self.cap = cv2.VideoCapture(0)
        self.frame_queue: BasicQueue = BasicQueue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _reader(self) -> None:
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)

    def read(self) -> np.ndarray:
        return self.frame_queue.get()
