# flake8: noqa

import argparse
import os.path
import sys
from typing import Dict

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(sys.modules[__name__].__file__), '..')))  # type: ignore  # noqa


import cv2

from frame.camera import (add_camera_parameters, get_codec,
                          parse_camera_settings, set_camera_parameters)


def parse_args() -> Dict:
    parser = argparse.ArgumentParser('Camera test.')
    parser = add_camera_parameters(parser)
    return vars(parser.parse_args())


def main(args: Dict) -> None:
    camera_settings = parse_camera_settings(args)
    cap = cv2.VideoCapture(camera_settings.input, camera_settings.api)
    set_camera_parameters(cap, camera_settings)

    print(cap.get(cv2.CAP_PROP_FOURCC))
    print('FPS: ', cap.get(cv2.CAP_PROP_FPS))
    print('Codec: ', get_codec(cap.get(cv2.CAP_PROP_FOURCC)))

    try:
        while True:
            ret, frame = cap.read()
            if ret:
                cv2.imshow('application', frame)
                if chr(cv2.waitKey(1) & 255) == 'q':
                    break
    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()
    cap.release()
    print('Closing')


if __name__ == '__main__':
    main(parse_args())
