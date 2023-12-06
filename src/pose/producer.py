from __future__ import annotations

import queue
import time
from multiprocessing import Process, Queue
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from frame.producer import FrameData
from frame.shared import FramePool
from ocsort.timer import Timer
from pipeline.data import (BaseData, CloseData, DataCollection,
                           pipeline_data_generator)
from pose.pose import BODY_POINTS, Pose
from segmentation.base import BodyPartSegmentation
from tracking.producer import TrackingData


class PoseData(BaseData):
    def __init__(
        self,
        landmarks: List[np.ndarray],
        raw_landmarks: List[Any],
    ) -> None:
        super().__init__()
        self.landmarks = landmarks
        self.raw_landmarks = raw_landmarks

    def get_landmarks_xy(
        self,
        id: int,
        specific_bodypart: BodyPartSegmentation = BodyPartSegmentation.ALL,
        visibility_threshold: float = 0.5
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not self.landmarks[id].any():
            return None, None
        point_modes = []
        landmarks = []
        for landmark_id, landmark in enumerate(self.landmarks[id]):
            if landmark[3] > visibility_threshold:
                if specific_bodypart != BodyPartSegmentation.ALL:
                    points = BODY_POINTS[specific_bodypart.value - 1]
                    if points[landmark_id] >= 0.0:
                        point_modes.append(points[landmark_id])
                        landmarks.append(self.landmarks[id][landmark_id, :2])
                else:
                    point_modes.append(1.0)
                    landmarks.append(self.landmarks[id][landmark_id, :2])
        return np.array(landmarks), point_modes


def produce_pose(
    input_queue: Queue[DataCollection],
    output_queue: Queue[DataCollection],
    model_complexity: int = 1,
    frame_pool: Optional[FramePool] = None
) -> None:
    reduce_frame_discard_timer = 0.0
    timer = Timer()
    pose = Pose(model_complexity)
    frame = 0
    for data in pipeline_data_generator(
        input_queue,
        output_queue,
        [TrackingData]
    ):
        timer.tic()
        image = data.get(FrameData).get_frame(frame_pool)
        image.flags.writeable = False
        conv_pose_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        all_landmarks = []
        all_raw_landmarks = []
        tracking_data = data.get(TrackingData)
        for id in range(len(tracking_data.targets)):
            pad_box = tracking_data.get_padded_box(id)
            cropped_conv_image = \
                conv_pose_image[int(pad_box[1]):int(pad_box[3]),
                                int(pad_box[0]):int(pad_box[2])]
            landmarks, raw_landmarks = pose.predict(cropped_conv_image)

            if landmarks.any():
                landmarks[:, 0] += pad_box[0]
                landmarks[:, 1] += pad_box[1]
            all_landmarks.append(landmarks)
            all_raw_landmarks.append(raw_landmarks)

        if not output_queue.empty():
            try:
                discarded_frame = output_queue.get_nowait()
                if frame_pool and discarded_frame.has(FrameData):
                    frame_pool.free_frame(
                        discarded_frame.get(FrameData).frame)
                reduce_frame_discard_timer += 0.015
            except queue.Empty:
                reduce_frame_discard_timer -= 0.001
                if reduce_frame_discard_timer < 0:
                    reduce_frame_discard_timer = 0

        output_queue.put(
            data.add(PoseData(all_landmarks, all_raw_landmarks)))
        timer.toc()
        frame += 1
        if frame == 100:
            timer.clear()
        if frame % 100 == 0 and frame > 100:
            print('Pose-FPS:', 1. / timer.average_time, 1. /
                  (timer.average_time + reduce_frame_discard_timer))
        if reduce_frame_discard_timer > 0.015:
            time.sleep(reduce_frame_discard_timer)
    pose.close()


class PoseProducer:
    def __init__(
        self,
        input_queue: Queue[DataCollection],
        output_queue: Queue[DataCollection],
        model_complexity: int = 1,
        frame_pool: Optional[FramePool] = None
    ) -> None:
        self.process: Optional[Process] = None
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.model_complexity = model_complexity
        self.frame_pool = frame_pool

    def start(self) -> None:
        self.process = Process(target=produce_pose, args=(
            self.input_queue,
            self.output_queue,
            self.model_complexity,
            self.frame_pool
        ))
        self.process.start()

    def stop(self) -> None:
        self.input_queue.put(DataCollection().add(CloseData()))
        if self.process:
            time.sleep(1)
            self.process.kill()
