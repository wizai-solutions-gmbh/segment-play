from __future__ import annotations

import queue
import time
from multiprocessing import Process, Queue
from multiprocessing.sharedctypes import Synchronized
from typing import List, Optional

import numpy as np

from frame.producer import FrameData
from frame.shared import FramePool
from ocsort.timer import Timer
from pipeline.data import (BaseData, CloseData, DataCollection,
                           pipeline_data_generator)
from pose.producer import PoseData
from segmentation.base import BodyPartSegmentation
from segmentation.mobile_sam import MobileSam
from segmentation.sam import Sam
from tracking.producer import TrackingData
from util.image import clip_section_xyxy, scale_image


class SegmentationData(BaseData):
    def __init__(
        self,
        masks: List[List[np.ndarray]],
        mask_scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.masks = masks
        self.mask_scale = mask_scale

    def get_box(self, id: int) -> np.ndarray:
        return self.targets[id][:4]

    def get_padded_box(self, id: int) -> np.ndarray:
        return self.targets[id][5:]


def produce_segmentation(
    input_queue: Queue[DataCollection],
    output_queue: Queue[DataCollection],
    down_scale: Optional[float] = None,
    fast: bool = True,
    frame_pool: Optional[FramePool] = None,
    specific_bodypart: Optional[Synchronized] = None
) -> None:
    reduce_frame_discard_timer = 0.0
    timer = Timer()
    segment = MobileSam() if fast else Sam()
    frame = 0
    for data in pipeline_data_generator(
        input_queue,
        output_queue,
        [TrackingData]
    ):
        timer.tic()
        scaled_image = data.get(FrameData).get_frame(frame_pool)
        if down_scale is not None:
            scaled_image = scale_image(scaled_image, 1.0 / down_scale)
        segment.set_image(scaled_image)
        segment.prepare_prompts(scaled_image)
        all_masks = []
        tracking_data = data.get(TrackingData)
        for id in range(len(tracking_data.targets)):
            input_box = tracking_data.get_box(id)
            pad_box = tracking_data.get_padded_box(id)
            if down_scale:
                input_box /= down_scale
                pad_box /= down_scale

            landmarks = None

            if data.has(PoseData):
                bodypart = None
                if specific_bodypart is not None:
                    bodypart = BodyPartSegmentation(specific_bodypart.value)
                landmarks, point_mode = data.get(
                    PoseData).get_landmarks_xy(id, bodypart)
                if landmarks is not None:
                    if down_scale:
                        landmarks /= down_scale

                if landmarks is not None and specific_bodypart is not None \
                        and specific_bodypart.value != BodyPartSegmentation.ALL.value:
                    padding = min(
                        input_box[2] - input_box[0],
                        input_box[3] - input_box[1]
                    )
                    padding *= 0.25
                    positions_x = [landmark[0] for landmark,
                                   pm in zip(landmarks, point_mode) if pm == 1]
                    positions_y = [landmark[1] for landmark,
                                   pm in zip(landmarks, point_mode) if pm == 1]
                    if positions_x and positions_y:
                        min_x = min(positions_x)
                        max_x = max(positions_x)
                        min_y = min(positions_y)
                        max_y = max(positions_y)
                        if input_box[0] < min_x - padding:
                            input_box[0] = max(min_x - padding, input_box[0])
                        if input_box[1] < min_y - padding:
                            input_box[1] = max(min_y - padding, input_box[1])
                        if input_box[2] > max_x + padding:
                            input_box[2] = min(max_x + padding, input_box[2])
                        if input_box[3] > max_y + padding:
                            input_box[3] = min(max_y + padding, input_box[3])
                        input_box[:4] = [*clip_section_xyxy(
                            input_box[0],
                            input_box[1],
                            input_box[2],
                            input_box[3],
                            scaled_image
                        )]

            new_mask = segment.bbox_masks(input_box, landmarks, point_mode)

            # mask potentially overlap the bounding box, therefore use
            # padded bounding box for cutting out the mask
            new_mask = new_mask[
                0,
                int(pad_box[1]):int(pad_box[3]),
                int(pad_box[0]):int(pad_box[2])
            ]
            if new_mask.shape[0] <= 0 or new_mask.shape[1] <= 0:
                print('New mask is empty', tracking_data.get_box(
                    id), tracking_data.get_padded_box(id))
            all_masks.append([new_mask])
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
        output_queue.put(data.add(SegmentationData(
            all_masks, down_scale)))
        timer.toc()
        frame += 1
        if frame == 100:
            timer.clear()
        if frame % 100 == 0 and frame > 100:
            print('Segmentation-FPS:', 1. / timer.average_time, 1. /
                  (timer.average_time + reduce_frame_discard_timer))
        if reduce_frame_discard_timer > 0.015:
            time.sleep(reduce_frame_discard_timer)


class SegmentProducer:
    def __init__(
        self,
        input_queue: Queue[DataCollection],
        output_queue: Queue[DataCollection],
        down_scale: Optional[float] = None,
        fast: bool = True,
        frame_pool: Optional[FramePool] = None,
        specific_bodypart: Optional[Synchronized[int]] = None
    ) -> None:
        self.process: Optional[Process] = None
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.down_scale = down_scale
        self.fast = fast
        self.frame_pool = frame_pool
        self.specific_bodypart = specific_bodypart

    def start(self) -> None:
        self.process = Process(target=produce_segmentation, args=(
            self.input_queue,
            self.output_queue,
            self.down_scale,
            self.fast,
            self.frame_pool,
            self.specific_bodypart
        ))
        self.process.start()

    def stop(self) -> None:
        self.input_queue.put(DataCollection().add(CloseData()))
        if self.process:
            time.sleep(1)
            self.process.kill()
