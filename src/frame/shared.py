from multiprocessing import Manager
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

from frame.camera import CameraSettings, set_camera_parameters


class FramePool:
    def __init__(self, template: np.ndarray, maxsize: int) -> None:
        self.dtype = template.dtype
        self.shape = template.shape
        self.byte_count = template.nbytes
        self.queue_manager = Manager()
        self.memory_manager = SharedMemoryManager()
        self.memory_manager.start()

        self.frame_pool: List[np.ndarray] = []
        self.shared_memory: List[SharedMemory] = []
        self.free_frames = self.queue_manager.Queue(maxsize)
        for index in range(maxsize):
            self.shared_memory.append(self.memory_manager.SharedMemory(
                self.byte_count))
            self.frame_pool.append(np.frombuffer(
                self.shared_memory[index].buf,
                dtype=self.dtype
            ).reshape(self.shape))
            self.free_frames.put(index)

    def __getstate__(self) -> Dict:
        d = dict(self.__dict__)
        if 'queue_manager' in d:
            del d['queue_manager']
        if 'memory_manager' in d:
            del d['memory_manager']
        if 'frame_pool' in d:
            del d['frame_pool']
        return d

    def __setstate__(self, d: Dict) -> None:
        shared_memories = d['shared_memory']
        d['frame_pool'] = [
            np.frombuffer(
                sm.buf,
                dtype=d['dtype'],
                count=d['byte_count']
            ).reshape(d['shape'])
            for sm in shared_memories
        ]
        # for array in d['frame_pool']:
        #   array.flags.writeable = False
        self.__dict__.update(d)

    def free_frame(self, index: int) -> None:
        self.free_frames.put(index)

    def put(self, frame: Union[np.ndarray, int]) -> int:
        if type(frame) is not np.ndarray:
            return frame
        index: int = self.free_frames.get()
        self.frame_pool[index][:] = frame[:]
        return index

    def get(self, index: int) -> np.ndarray:
        return self.frame_pool[index]

    def close(self) -> None:
        self.memory_manager.shutdown()


def create_frame_pool(
    maxsize: int,
    settings: Optional[CameraSettings] = None
) -> FramePool:
    if not settings:
        settings = CameraSettings()
    cap = cv2.VideoCapture(settings.input, settings.api)
    set_camera_parameters(cap, settings)

    while True:
        ret, frame = cap.read()
        if ret:
            frame_pool = FramePool(frame, maxsize)
            cap.release()
            return frame_pool
