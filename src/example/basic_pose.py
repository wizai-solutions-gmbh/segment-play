# flake8: noqa

from __future__ import annotations

import argparse
import os.path
import sys
from typing import Dict

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(sys.modules[__name__].__file__), '..')))  # type: ignore  # noqa

import queue
from multiprocessing import Queue, freeze_support

import cv2

from frame.camera import add_camera_parameters, parse_camera_settings
from frame.producer import FrameData, VideoCaptureProducer
from frame.shared import create_frame_pool
from ocsort.timer import Timer
from pipeline.data import DataCollection
from pose.pose import Pose, PoseRenderer
from tracking.producer import TrackingData, TrackProducer
from util.visualize import show_box


def parse_args() -> Dict:
    parser = argparse.ArgumentParser(
        'Basic hide people via segmentation and background estimation.')
    parser.add_argument('--down-scale', type=float,
                        default=1.0, help='Downscale rate')
    parser = add_camera_parameters(parser)
    return vars(parser.parse_args())


def main(args: Dict) -> None:
    down_scale = args.get('down_scale', 1.0)
    camera_settings = parse_camera_settings(args)

    frame_queue: Queue[DataCollection] = Queue()
    frame_pool = create_frame_pool(100, camera_settings)

    tracking_queue: Queue[DataCollection] = Queue()
    tracker = TrackProducer(frame_queue, tracking_queue,
                            down_scale, frame_pool)

    timer = Timer()
    cap = VideoCaptureProducer(frame_queue, camera_settings, frame_pool)
    cap.start()
    tracker.start()

    pose = Pose()
    pose_renderer = PoseRenderer()

    try:
        while True:
            try:
                data = tracking_queue.get(timeout=0.01)
                assert data.has(TrackingData)

                timer.tic()

                image = data.get(FrameData).get_frame(frame_pool)

                pose_image = image.copy()
                conv_pose_image = cv2.cvtColor(
                    pose_image, cv2.COLOR_BGR2RGB)

                tracking_data = data.get(TrackingData)
                for id in range(len(tracking_data.targets)):
                    pad_box = tracking_data.get_padded_box(id)
                    cropped_image = pose_image[int(pad_box[1]):int(pad_box[3]),
                                               int(pad_box[0]):int(pad_box[2])]
                    cropped_conv_image = conv_pose_image[int(pad_box[1]):int(pad_box[3]),
                                                         int(pad_box[0]):int(pad_box[2])]
                    landmarks, raw_landmarks = pose.predict(cropped_conv_image)

                    filtered_landmarks = [
                        landmark
                        for landmark in landmarks
                        if landmark[3] > 0.5
                    ]
                    for landmark in filtered_landmarks:
                        cropped_image = cv2.circle(
                            cropped_image,
                            (int(landmark[0]), int(landmark[1])),
                            radius=5,
                            color=(
                                0, 255 if landmark[3] >= 0.5 else 0, 255 if landmark[3] < 0.5 else 0),
                            thickness=10
                        )
                    pose_image[int(pad_box[1]):int(pad_box[3]),
                               int(pad_box[0]):int(pad_box[2])] = cropped_image

                    if raw_landmarks:
                        pose_renderer.draw(
                            pose_image,
                            raw_landmarks,
                            (int(pad_box[0]), int(pad_box[1])),
                            (cropped_image.shape[1] / pose_image.shape[1],
                             cropped_image.shape[0] / pose_image.shape[0]))

                for id in range(len(tracking_data.targets)):
                    input_box = tracking_data.get_box(id)
                    pose_image = show_box(pose_image, input_box)

                cv2.imshow('application', pose_image)
                if frame_pool:
                    frame_pool.free_frame(data.get(FrameData).frame)

                if chr(cv2.waitKey(1) & 255) == 'q':
                    break

                timer.toc()
            except queue.Empty:
                # print("Consumer:Timeout reading from the track Queue")
                pass
    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()
    cap.stop()
    tracker.stop()
    pose.close()
    print('Closing')


if __name__ == '__main__':
    freeze_support()
    main(parse_args())
