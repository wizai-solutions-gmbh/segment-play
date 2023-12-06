# flake8: noqa

import argparse
import os.path
import sys
from typing import Dict

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(sys.modules[__name__].__file__), '..')))  # type: ignore  # noqa
import cv2
import numpy as np

from background import Background
from frame.camera import (add_camera_parameters, parse_camera_settings,
                          set_camera_parameters)
from segmentation.mobile_sam import MobileSam
from segmentation.sam import Sam
from util.image import scale_image
from util.mask import apply_mask, scale_mask
from util.visualize import show_box


def parse_args() -> Dict:
    parser = argparse.ArgumentParser(
        'Hide segmented mask for whatever enters a configurable box.')
    parser.add_argument('--slow', dest='slow', default=False,
                        action='store_true', help='Using faster SAM model.')
    parser.add_argument('--down-scale', type=float,
                        default=1.0, help='Downscale rate')
    parser.add_argument('--x', type=float, default=0.25,
                        help='x position in percentage of width')
    parser.add_argument('--y', type=float, default=0.25,
                        help='y position in percentage of height')
    parser.add_argument('--x2', type=float, default=0.75,
                        help='x2 position in percentage of width')
    parser.add_argument('--y2', type=float, default=0.75,
                        help='y2 position in percentage of height')
    parser = add_camera_parameters(parser)

    return vars(parser.parse_args())


def main(args: Dict) -> None:
    fast = not args.get('slow', True)
    down_scale = args.get('down_scale', 1.0)

    x = args.get('x', 0.25)
    y = args.get('y', 0.25)
    x2 = args.get('x2', 0.25)
    y2 = args.get('y2', 0.25)

    segment = MobileSam() if fast else Sam()
    background = Background()

    camera_settings = parse_camera_settings(args)
    cap = cv2.VideoCapture(camera_settings.input, camera_settings.api)
    set_camera_parameters(cap, camera_settings)

    try:
        while True:
            retval, image = cap.read()
            if not retval:
                continue

            if background.avg is None:
                background.add_black(image.shape)

            scaled_image = scale_image(image, 1 / down_scale)

            segment.set_image(scaled_image)

            width = scaled_image.shape[1]
            height = scaled_image.shape[0]
            input_box = np.array(
                [width * x, height * y, width * x2, height * y2])
            segment.prepare_prompts(scaled_image)
            masks = segment.bbox_masks(input_box)
            mask = masks[0]
            mask = scale_mask(mask, down_scale)

            mask_image = apply_mask(image, background.get_bg(), mask)
            background.add_frame(mask_image)

            boxed_image = show_box(mask_image, input_box / down_scale)

            cv2.imshow('frame', boxed_image)
            if chr(cv2.waitKey(1) & 255) == 'q':
                break
    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(parse_args())
