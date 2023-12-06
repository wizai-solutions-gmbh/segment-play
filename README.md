# Segment Play

Segment Play is a tool that shows some possibilities of real-time segmentation in an interactive scenario with a choreographic or game-like context.

## Install

```
pip install -r requirements.txt
```

Install pytorch see (https://pytorch.org/get-started/locally/).
Following version were used in development on Windows 10 & 11:

```
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
```

### Models

Segmentation model for MobileSAM can be downloaded from [here](https://github.com/ChaoningZhang/MobileSAM) or directly [mobile-sam.pt](https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt).

Segmentation models for SAM can be downloaded from [here](https://github.com/CASIA-IVA-Lab/FastSAM) and is licensed under the [Apache 2.0 license](https://github.com/facebookresearch/segment-anything/blob/main/LICENSE).

Models used for tracking can be downloaded [here](https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/ONNXRuntime) or directly [YOLOX-Tiny](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx)

## Demo

Starting `python src/scenario_director.py` will run a demo using an USB-Camera input. The application will detect and track any person entering the cameras field of view. Various effects can be applied via segmentation of the detected and tracked persons in near real-time. You can choose the active effects by pressing certain keys, the controls are listed in the terminal. Some effects react to different attributes (e.g. position in the room and shape) of a person and allowing a live interaction with the application. The persons interacting, can see the effect immediately on a connected display. I recommend placing the webcam above a large connected display facing the area of interest.

### Hardware

* Webcam

Recommended:

* NVIDIA GeForce RTX 3060 or better
* CUDA
* Large display to see the live-image from a distance

If you are interested in a custom solution and hardware, feel free to contact [wizAI](mailto:info@wizai.com?subject=[GitHub]%20Segment-Play)

## License

This project is released under the Apache 2.0 license (see [LICENSE](LICENSE)).
Portions of the Code are based and taken from [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) and [OCSort](https://github.com/noahcao/OC_SORT) located in `src/ocsort`. These portions have their License information included in their files.

## Acknowledgement

This project was initially created by Julian Rogawski during the [Choreographic Coding Lab (CCL)](https://choreographiccoding.org) 06.-10.09.2023 hosted by [Motion Bank](https://motionbank.org/) the German Sports University Cologne at the Institute of Dance and Movement Culture. The CCL is part of the research project https://vortanz.ai with [wizAI solutions GmbH](https://wizai.com/) as a partner. Special thanks to the hosts and participants of the CCL 2023 for all the inspiration.
