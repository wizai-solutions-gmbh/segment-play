# flake8: noqa
# type: ignore

"""
Script taken from https://github.com/noahcao/OC_SORT@7a390a5:
MIT License

Copyright (c) 2021 Yifu Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Modifications copyright (C) 2023 Julian Rogawski and wizAI solutions GmbH
"""


import numpy as np
import onnxruntime

from ocsort.preprocess import preproc as preprocess
from ocsort.utils import demo_postprocess, multiclass_nms


class Predictor(object):
    def __init__(self, args):
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.args = args
        self.session = onnxruntime.InferenceSession(
            args.model, providers=['CPUExecutionProvider'])
        self.input_shape = tuple(map(int, args.input_shape.split(',')))

    def inference(self, ori_img, timer):
        img_info = {'id': 0}
        height, width = ori_img.shape[:2]
        img_info['height'] = height
        img_info['width'] = width
        img_info['raw_img'] = ori_img

        # , self.rgb_means, self.std)
        img, ratio = preprocess(ori_img, self.input_shape, mean=None, std=None)
        img_info['ratio'] = ratio
        ort_inputs = {self.session.get_inputs()[0].name: img[None, :, :, :]}
        timer.tic()
        output = self.session.run(None, ort_inputs)
        predictions = demo_postprocess(
            output[0], self.input_shape, p6=self.args.with_p6)[0]

        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(
            boxes_xyxy, scores, nms_thr=self.args.nms_thr, score_thr=self.args.score_thr)
        if dets is not None:
            return dets[:, :-1], img_info
        else:
            return None, img_info
