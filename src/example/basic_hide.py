# flake8: noqa

import argparse
import os.path
import sys
from typing import Dict

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(sys.modules[__name__].__file__), '..')))  # type: ignore  # noqa

import queue
from multiprocessing import Queue, freeze_support

import cv2

from background import Background
from frame.camera import add_camera_parameters, parse_camera_settings
from frame.producer import FrameData, VideoCaptureProducer
from frame.shared import create_frame_pool
from ocsort.timer import Timer
from pipeline.data import DataCollection
from segmentation.mobile_sam import MobileSam
from segmentation.sam import Sam
from tracking.producer import TrackingData, TrackProducer
from util.image import scale_image
from util.mask import add_masks, apply_mask, dilate, scale_mask
from util.visualize import show_box


def parse_args() -> Dict:
    parser = argparse.ArgumentParser(
        'Basic hide people via segmentation and background estimation.')
    parser.add_argument('--slow', dest='slow', default=False,
                        action='store_true', help='Using faster SAM model.')
    parser.add_argument('--down-scale', type=float,
                        default=1.0, help='Downscale rate')
    parser = add_camera_parameters(parser)
    return vars(parser.parse_args())


def main(args: Dict) -> None:
    fast = not args.get('slow', True)
    down_scale = args.get('down_scale', 1.0)
    camera_settings = parse_camera_settings(args)

    segment = MobileSam() if fast else Sam()
    background = Background()

    frame_queue: Queue[DataCollection] = Queue()
    frame_pool = create_frame_pool(100, camera_settings)

    tracking_queue: Queue[DataCollection] = Queue()
    tracker = TrackProducer(frame_queue, tracking_queue,
                            down_scale, frame_pool)

    timer = Timer()
    cap = VideoCaptureProducer(frame_queue, camera_settings, frame_pool)
    cap.start()
    tracker.start()

    try:
        while True:
            try:
                data = tracking_queue.get(timeout=0.01)
                assert data.has(TrackingData)

                timer.tic()

                image = data.get(FrameData).get_frame(frame_pool)
                tracking_data = data.get(TrackingData)

                if background.avg is None:
                    background.add_black(image.shape)

                scaled_image = scale_image(image, 1.0 / down_scale)
                segment.set_image(scaled_image)
                segment.prepare_prompts(scaled_image)

                hide_mask = None
                for id in range(len(tracking_data.targets)):
                    input_box = tracking_data.get_box(id)
                    masks = segment.bbox_masks(input_box / down_scale)
                    mask = masks[0]
                    if hide_mask is None:
                        hide_mask = mask
                    else:
                        hide_mask = add_masks(hide_mask, mask)

                if hide_mask is not None:
                    hide_mask = scale_mask(hide_mask, down_scale)
                    hide_mask = dilate(hide_mask)
                    image = apply_mask(image, background.get_bg(), hide_mask)

                background.add_frame(image)

                for id in range(len(tracking_data.targets)):
                    input_box = tracking_data.get_box(id)
                    image = show_box(image, input_box)

                cv2.imshow('application', image)
                if frame_pool:
                    frame_pool.free_frame(data.get(FrameData).frame)

                if chr(cv2.waitKey(1) & 255) == 'q':
                    break

                timer.toc()
            except queue.Empty:
                pass  # print("Consumer:Timeout reading from the track Queue")
    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()
    cap.stop()
    tracker.stop()
    print('Closing')


if __name__ == '__main__':
    freeze_support()
    main(parse_args())
