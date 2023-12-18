import argparse
import random
import time
from multiprocessing import Value, freeze_support
from multiprocessing.sharedctypes import Synchronized
from typing import Dict, Optional

import cv2
import numpy as np

from background import Background
from frame.camera import (CameraSettings, add_camera_parameters,
                          parse_camera_settings)
from frame.producer import FrameData
from frame.shared import FramePool, create_frame_pool
from input import Interaction
from ocsort.timer import Timer
from pipeline.data import DataCollection, ExceptionCloseData
from pipeline.manager import FrameProcessingPipeline
from pipeline.stats import TrackFrameStats
from pose.pose import PoseRenderer
from pose.producer import PoseData
from segmentation.base import BodyPartSegmentation
from segmentation.producer import SegmentationData
from settings import GameSettings
from tracking.producer import TrackingData
from util.image import create_black_image
from util.mask import (add_masks, apply_mask, apply_mask_grayscale, dilate,
                       scale_mask)
from util.visualize import show_box


def parse_args() -> Dict:
    parser = argparse.ArgumentParser(
        'Demo controllable via keys to control various effects.')
    parser.add_argument('--fullscreen', dest='fullscreen', default=False,
                        action='store_true', help='Show result in fullscreen mode.')
    parser.add_argument('--slow', dest='slow', default=False,
                        action='store_true', help='Using faster SAM model.')
    parser.add_argument('--down-scale', type=float,
                        default=1.0, help='Downscale rate')
    parser.add_argument('--segment-processes', type=int, default=2,
                        help='Number of processes for segmentation.')
    parser.add_argument('--save', dest='save', default=False,
                        action='store_true', help='Save images for every processed frame, with original image.')  # noqa: E501
    parser = add_camera_parameters(parser)

    return vars(parser.parse_args())


def get_verticality(vector: np.ndarray) -> float:
    normalize_vector = vector / np.sqrt(np.sum(vector**2))
    return normalize_vector[1]


class Director:
    def __init__(
            self,
            settings: GameSettings,
            segment_processes: int = 2,
            fast: bool = True,
            down_scale: Optional[float] = None,
            camera_settings: Optional[CameraSettings] = None,
            frame_pool: Optional[FramePool] = None,
            fullscreen: bool = False
    ) -> None:
        self.bodypart_segmentation: Synchronized[int] = Value(
            'i', BodyPartSegmentation.ALL.value)  # type: ignore
        self.stats: TrackFrameStats = TrackFrameStats(frame_pool)
        self.background: Background = Background()
        self.settings: GameSettings = settings
        self.interaction: Interaction = Interaction()
        self.processor: FrameProcessingPipeline = FrameProcessingPipeline(
            segment_processes,
            down_scale,
            fast,
            camera_settings,
            frame_pool,
            self.bodypart_segmentation
        )
        self.frame_pool = frame_pool
        self.pose_renderer = PoseRenderer()

        if fullscreen:
            cv2.namedWindow('application', cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(
                'application', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def frame(self, data: DataCollection) -> np.ndarray:
        original_image = data.get(FrameData).get_frame(self.frame_pool)
        image = original_image.copy()

        if self.background.avg is None:
            self.background.add_black(image.shape)

        hide_mask = None
        mirror_masks = []
        tracking_data = data.get(TrackingData)
        segmentation_data = data.get(SegmentationData)
        for id, masks in enumerate(segmentation_data.masks):
            pad_box = tracking_data.get_padded_box(id).astype(np.int32)
            input_box = tracking_data.get_box(id).astype(np.int32)
            track_id = tracking_data.get_tracking_id(id)
            if track_id not in self.settings.id_position_map.keys():
                self.settings.id_position_map[track_id] = random.random()

            if len(masks) == 0:
                continue

            mask = masks[0]

            mirror_p = int(
                track_id) % 2 == 0 and self.settings.random_people_mirror
            visible = True
            ratio = (input_box[3] - input_box[1]) / \
                (input_box[2] - input_box[0])
            if self.settings.form_invisibility:
                if data.has(PoseData) and data.get(
                        PoseData).raw_landmarks[id] is not None:
                    pose_landmarks = data.get(
                        PoseData).raw_landmarks[id].landmark
                    upper_right_arm_vec = np.array([
                        pose_landmarks[14].x - pose_landmarks[12].x,
                        pose_landmarks[14].y - pose_landmarks[12].y,
                        (pose_landmarks[14].z - pose_landmarks[12].z) * 0.0,
                    ], dtype=float)
                    lower_right_arm_vec = np.array([
                        pose_landmarks[16].x - pose_landmarks[14].x,
                        pose_landmarks[16].y - pose_landmarks[14].y,
                        (pose_landmarks[16].z - pose_landmarks[14].z) * 0.0,
                    ], dtype=float)
                    upper_left_arm_vec = np.array([
                        pose_landmarks[13].x - pose_landmarks[11].x,
                        pose_landmarks[13].y - pose_landmarks[11].y,
                        (pose_landmarks[13].z - pose_landmarks[11].z) * 0.0,
                    ], dtype=float)
                    lower_left_arm_vec = np.array([
                        pose_landmarks[15].x - pose_landmarks[13].x,
                        pose_landmarks[15].y - pose_landmarks[13].y,
                        (pose_landmarks[15].z - pose_landmarks[13].z) * 0.0,
                    ], dtype=float)
                    if get_verticality(upper_right_arm_vec) \
                            + get_verticality(lower_right_arm_vec) \
                            + get_verticality(upper_left_arm_vec) \
                            + get_verticality(lower_left_arm_vec) > 0.8 * 4:
                        visible = False
                elif ratio > 2.5:
                    visible = False

            width = pad_box[2] - pad_box[0]
            center_x = (input_box[0] + input_box[2]) / 2.0
            mirror_masks.append((
                mask,
                mirror_p,
                pad_box,
                visible,
                center_x,
                width,
                track_id
            ))
            mask_scale = 1.0
            if segmentation_data.mask_scale:
                mask_scale = segmentation_data.mask_scale
            if hide_mask is None:
                hide_mask = np.zeros((
                    int(image.shape[0] / mask_scale),
                    int(image.shape[1] / mask_scale)
                ), dtype=bool)
            hide_mask = add_masks(
                hide_mask,
                mask,
                (int(pad_box[1] / mask_scale),
                 int(pad_box[0] / mask_scale))
            )

        if hide_mask is not None:
            if segmentation_data.mask_scale:
                hide_mask = scale_mask(hide_mask, segmentation_data.mask_scale)
            hide_mask = dilate(hide_mask)
            image = apply_mask(image, self.background.get_bg(), hide_mask)

        self.background.add_frame(image)

        if not self.settings.all_invisibility:
            if self.settings.hide_background:
                image = np.zeros(image.shape, dtype=np.uint8)
            flipped_original_image = cv2.flip(original_image, 1)
            if mirror_masks is not None:
                mirror_masks.sort(key=lambda x: x[1])
            for mirror_mask_data in mirror_masks:
                scale_m, mirror, box, visible, x_pos, p_width, track_id \
                    = mirror_mask_data
                if visible:
                    if segmentation_data.mask_scale:
                        scale_m = scale_mask(
                            scale_m, segmentation_data.mask_scale)
                    if not self.settings.hide_background:
                        scale_m = dilate(scale_m)
                    gray = False
                    if self.settings.gray_game:
                        color_pos = image.shape[1] * \
                            self.settings.id_position_map[track_id] * 0.5
                        mod_x_pos = int(x_pos) % int(
                            image.shape[1] * 0.5)
                        if abs(mod_x_pos - color_pos) > p_width:
                            gray = True
                    masking_image = original_image if not mirror \
                        else flipped_original_image
                    gray_mask = scale_m if not mirror else np.fliplr(scale_m)
                    x_pos = box[0]
                    if mirror:
                        x_pos = (image.shape[1] - x_pos) - p_width

                    image = apply_mask_grayscale(
                        image,
                        masking_image,
                        gray_mask,
                        gray,
                        (box[1], x_pos)
                    )

        if self.settings.show_poses:
            if data.has(PoseData):
                pose_data = data.get(PoseData)
                for id in range(len(tracking_data.targets)):
                    if pose_data.raw_landmarks[id]:
                        input_box = tracking_data.get_padded_box(id)
                        image = self.pose_renderer.draw(
                            image,
                            pose_data.raw_landmarks[id],
                            (int(input_box[0]), int(input_box[1])),
                            ((input_box[2] - input_box[0]) / image.shape[1],
                             (input_box[3] - input_box[1]) / image.shape[0])
                        )

        if self.settings.show_boxes:
            for id in range(len(tracking_data.targets)):
                input_box = tracking_data.get_box(id)
                track_id = tracking_data.get_tracking_id(id)
                image = show_box(image, input_box, track_id)

        if self.settings.overall_mirror:
            image = cv2.flip(image, 1)

        if self.settings.black:
            image = create_black_image(image.shape)

        return image

    def run(self) -> None:
        self.processor.start()

        overall_timer = Timer()
        overall_timer.tic()
        timer = Timer()
        start_time = time.time()
        recording_base_name = f'recordings/{str(int(start_time))}_'
        frame_count = 0

        for data in self.processor.get_frames():
            if data.has(ExceptionCloseData):
                print('Closing because of an exception in the pipeline!')
                print(data.get(ExceptionCloseData).exception)
                break
            elif data.is_closed():
                print('Closing package received.')
                break
            timer.tic()

            processed_image = self.frame(data)
            self.stats.add(data)

            if self.settings.save_imgs:
                recording_original_name = \
                    f'{recording_base_name}_processed_{frame_count}.png'
                cv2.imwrite(recording_original_name, processed_image)
                recording_processed_name = \
                    f'{recording_base_name}_original_{frame_count}.png'
                cv2.imwrite(recording_processed_name,
                            data.get(FrameData).get_frame(self.frame_pool))

            cv2.imshow('application', processed_image)

            if self.frame_pool:
                self.frame_pool.free_frame(data.get(FrameData).frame)

            frame_count += 1
            key = chr(cv2.waitKey(1) & 255)
            if key == 'q':
                break
            self.settings.handle_key(key)
            seg_change, seg_setting = self.settings.check_segmentation()
            if seg_change:
                self.bodypart_segmentation.value = seg_setting

            timer.toc()
            if frame_count % 100 == 0:
                print('Mask-FPS:', 1. / timer.average_time)

            overall_timer.toc()
            if frame_count == 50:
                timer.clear()
            if frame_count % 50 == 0 and frame_count > 50:
                print('Overall-FPS: ', 1. / overall_timer.average_time)
                print('Processing delay: ', self.stats.get_avg_delay())
                if self.frame_pool:
                    print('Avg frame processing: ',
                          self.stats.get_processing_frames())
            overall_timer.tic()

    def stop(self) -> None:
        self.processor.stop()
        cv2.destroyAllWindows()


def main(args: Dict) -> None:
    settings = GameSettings()
    settings.print()
    settings.save_imgs = args.get('save', False)
    camera_settings = parse_camera_settings(args)
    frame_pool = create_frame_pool(30, camera_settings)

    director = Director(
        settings,
        args.get('segment_processes', 2),
        not args.get('slow', False),
        args.get('down_scale', False),
        camera_settings,
        frame_pool,
        args.get('fullscreen', False)
    )

    try:
        director.run()
    except KeyboardInterrupt:
        pass

    director.stop()


if __name__ == '__main__':
    freeze_support()
    main(parse_args())
