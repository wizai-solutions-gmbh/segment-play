from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import onnxruntime

from ocsort.ocsort import OCSort
from ocsort.onnx_inference import demo_postprocess, multiclass_nms, preprocess
from util.image import clip_section


@dataclass
class TrackObject:
    appearance_count: int = 0
    last_appearance: List[int] = field(
        default_factory=lambda: [0, 0, 0, 0, 0])
    last_frame: int = 0


class Tracker:
    def __init__(self, down_scale: float = 1.0) -> None:
        self.session = onnxruntime.InferenceSession(
            'models/yolox_tiny.onnx', providers=['CPUExecutionProvider'])
        self.input_shape = (416, 416)
        self.nms_thr = 0.7
        self.score_thr = 0.1
        self.min_box_area = 10
        self.ocsort = OCSort(det_thresh=0.6, iou_threshold=0.3)
        self.current_targets: List[List[int]] = []
        self.track_objects: Dict[int, TrackObject] = {}
        self.current_frame = 0
        self.down_scale = down_scale

    def inference(
        self,
        image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Dict[str, Union[int, np.ndarray]]]:
        img_info: Dict[str, Union[int, np.ndarray]] = {'id': 0}
        height, width = image.shape[:2]
        img_info['height'] = height
        img_info['width'] = width
        img_info['raw_img'] = image

        img, ratio = preprocess(image, self.input_shape, mean=None, std=None)
        img_info['ratio'] = ratio
        ort_inputs = {self.session.get_inputs()[0].name: img[None, :, :, :]}

        output = self.session.run(None, ort_inputs)
        predictions = demo_postprocess(
            output[0], self.input_shape, p6=False)[0]

        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio
        detections = multiclass_nms(
            boxes_xyxy,
            scores,
            nms_thr=self.nms_thr,
            score_thr=self.score_thr
        )

        if detections is not None:
            filtered_dets = detections[detections[:, 5] == 0]
            return filtered_dets[:, :-1], img_info
        else:
            return None, img_info

    def update(self, image: np.ndarray) -> None:
        outputs, img_info = self.inference(image)
        online_targets = self.ocsort.update(
            outputs,
            [img_info['height'],
             img_info['width']], [
                img_info['height'], img_info['width']])
        current_ids = []
        for target in online_targets:
            padding = max(target[3] * 0.1, target[2] * 0.1)
            padded_box = [
                target[0] - padding,
                target[1] - padding,
                target[2] + padding,
                target[3] + padding,
            ]
            if self.down_scale != 1.0:
                padded_box = self.prepare_scale_bb(
                    int(self.down_scale), padded_box, image)
                target = self.prepare_scale_bb(
                    int(self.down_scale), target, image)
            else:
                padded_box = self.clip_bb(padded_box, image)
                target = self.clip_bb(target, image)
            target = np.append(target, padded_box)
            tlhw = [int(target[0]), int(target[1]), int(target[2]) -
                    int(target[0]), int(target[3]) - int(target[1])]
            tid = int(target[4])
            vertical = tlhw[2] / tlhw[3] > 1.6
            if tlhw[2] * tlhw[3] > self.min_box_area and not vertical:
                current_ids.append(tid)
                if tid in self.track_objects.keys():
                    self.track_objects[tid].appearance_count += 1
                    self.track_objects[tid].last_frame = self.current_frame
                    self.track_objects[tid].last_appearance = target
                else:
                    self.track_objects[tid] = TrackObject(
                        1, target, self.current_frame)
        self.current_targets = []
        for _, track_object in self.track_objects.items():
            if self.current_frame - track_object.last_frame < 15:
                self.current_targets.append(track_object.last_appearance)

        self.current_frame += 1

    def prepare_scale_bb(
        self,
        divisor: int,
        tracking_data: np.ndarray,
        image: np.ndarray
    ) -> np.ndarray:
        tracking_data[0] = int(tracking_data[0]) - \
            int(tracking_data[0]) % divisor
        tracking_data[1] = int(tracking_data[1]) - \
            int(tracking_data[1]) % divisor
        tracking_data[2] = int(tracking_data[2]) + \
            (divisor - int(tracking_data[2]) % divisor)
        tracking_data[3] = int(tracking_data[3]) + \
            (divisor - int(tracking_data[3]) % divisor)
        return self.clip_bb(tracking_data, image)

    def clip_bb(
        self,
        tracking_data: np.ndarray,
        image: np.ndarray
    ) -> np.ndarray:
        x, y, w, h = clip_section(
            int(tracking_data[0]),
            int(tracking_data[1]),
            int(tracking_data[2] - tracking_data[0]),
            int(tracking_data[3] - tracking_data[1]),
            image
        )
        tracking_data[0] = x
        tracking_data[1] = y
        tracking_data[2] = x + w
        tracking_data[3] = y + h
        return tracking_data

    def get_all_targets(self) -> List[np.ndarray]:
        return self.current_targets
