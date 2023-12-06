import queue
from multiprocessing import Queue
from multiprocessing.sharedctypes import Synchronized
from typing import Generator, List, Optional

from frame.camera import CameraSettings
from frame.producer import VideoCaptureProducer
from frame.shared import FramePool
from pipeline.data import DataCollection
from pose.producer import PoseProducer
from segmentation.producer import SegmentProducer
from tracking.producer import TrackProducer


class FrameProcessingPipeline:
    def __init__(
        self,
        segment_processes: int = 2,
        down_scale: Optional[float] = None,
        fast: bool = True,
        camera_settings: Optional[CameraSettings] = None,
        frame_pool: Optional[FramePool] = None,
        specific_bodypart: Optional[Synchronized] = None
    ) -> None:
        self.frame_queue: Queue[DataCollection] = Queue()
        self.tracking_queue: Queue[DataCollection] = Queue()
        self.pose_queue: Queue[DataCollection] = Queue()  # optional
        self.segment_queue: Queue[DataCollection] = Queue()
        self.segments: List[SegmentProducer] = [
            SegmentProducer(
                self.pose_queue,  # tracking queue
                self.segment_queue,
                down_scale,
                fast,
                frame_pool,
                specific_bodypart
            )
            for _ in range(segment_processes)
        ]
        self.pose: PoseProducer = PoseProducer(
            self.tracking_queue, self.pose_queue, frame_pool=frame_pool)  # optional
        self.tracker: TrackProducer = TrackProducer(
            self.frame_queue, self.tracking_queue, down_scale, frame_pool)
        self.cap = VideoCaptureProducer(
            self.frame_queue, camera_settings, frame_pool)

    def start(self) -> None:
        self.cap.start()
        self.tracker.start()
        self.pose.start()  # optional
        for segment in self.segments:
            segment.start()

    def get_frames(self) -> Generator[DataCollection, None, None]:
        while True:
            try:
                yield self.segment_queue.get(timeout=0.01)
            except queue.Empty:
                pass  # print("Consumer:Timeout reading from the track Queue")

    def stop(self) -> None:
        self.cap.stop()
        self.tracker.stop()
        self.pose.stop()  # optional
        for segment in self.segments:
            segment.stop()
