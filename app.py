#!/usr/bin/env python

from __future__ import annotations

import functools
import os
import pathlib
import shlex
import subprocess
import sys
import urllib.request

if os.environ.get('SYSTEM') == 'spaces':
    import mim
    mim.install('mmcv-full==1.4', is_yes=True)

    subprocess.call(shlex.split('pip uninstall -y opencv-python'))
    subprocess.call(shlex.split('pip uninstall -y opencv-python-headless'))
    subprocess.call(
        shlex.split('pip install opencv-python-headless==4.5.5.64'))
    subprocess.call(shlex.split('pip install terminaltables==3.1.0'))
    subprocess.call(shlex.split('pip install mmpycocotools==12.0.3'))

    subprocess.call(shlex.split('pip install insightface==0.6.2'))
    subprocess.call(shlex.split('sed -i 23,26d __init__.py'),
                    cwd='insightface/detection/scrfd/mmdet')

import cv2
import gradio as gr
import huggingface_hub
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, 'insightface/detection/scrfd')

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

TITLE = 'insightface Face Detection (SCRFD)'
DESCRIPTION = 'This is an unofficial demo for https://github.com/deepinsight/insightface/tree/master/detection/scrfd.'

HF_TOKEN = os.getenv('HF_TOKEN')


def load_model(model_size: str, device) -> nn.Module:
    ckpt_path = huggingface_hub.hf_hub_download(
        'hysts/insightface',
        f'models/scrfd_{model_size}/model.pth',
        use_auth_token=HF_TOKEN)
    scrfd_dir = 'insightface/detection/scrfd'
    config_path = f'{scrfd_dir}/configs/scrfd/scrfd_{model_size}.py'
    model = init_detector(config_path, ckpt_path, device.type)
    return model


def update_test_pipeline(model: nn.Module, mode: int):
    cfg = model.cfg
    pipelines = cfg.data.test.pipeline
    for pipeline in pipelines:
        if pipeline.type == 'MultiScaleFlipAug':
            if mode == 0:  # 640 scale
                pipeline.img_scale = (640, 640)
                if hasattr(pipeline, 'scale_factor'):
                    del pipeline.scale_factor
            elif mode == 1:  # for single scale in other pages
                pipeline.img_scale = (1100, 1650)
                if hasattr(pipeline, 'scale_factor'):
                    del pipeline.scale_factor
            elif mode == 2:  # original scale
                pipeline.img_scale = None
                pipeline.scale_factor = 1.0
            transforms = pipeline.transforms
            for transform in transforms:
                if transform.type == 'Pad':
                    if mode != 2:
                        transform.size = pipeline.img_scale
                        if hasattr(transform, 'size_divisor'):
                            del transform.size_divisor
                    else:
                        transform.size = None
                        transform.size_divisor = 32


def detect(image: np.ndarray, model_size: str, mode: int,
           face_score_threshold: float,
           detectors: dict[str, nn.Module]) -> np.ndarray:
    model = detectors[model_size]
    update_test_pipeline(model, mode)

    # RGB -> BGR
    image = image[:, :, ::-1]
    preds = inference_detector(model, image)
    boxes = preds[0]

    res = image.copy()
    for box in boxes:
        box, score = box[:4], box[4]
        if score < face_score_threshold:
            continue
        box = np.round(box).astype(int)

        line_width = max(2, int(3 * (box[2:] - box[:2]).max() / 256))
        cv2.rectangle(res, tuple(box[:2]), tuple(box[2:]), (0, 255, 0),
                      line_width)

    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    return res


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_sizes = [
    '500m',
    '1g',
    '2.5g',
    '10g',
    '34g',
]
detectors = {
    model_size: load_model(model_size, device=device)
    for model_size in model_sizes
}
modes = [
    '(640, 640)',
    '(1100, 1650)',
    'original',
]

func = functools.partial(detect, detectors=detectors)

image_path = pathlib.Path('selfie.jpg')
if not image_path.exists():
    url = 'https://raw.githubusercontent.com/peiyunh/tiny/master/data/demo/selfie.jpg'
    urllib.request.urlretrieve(url, image_path)
examples = [[image_path.as_posix(), '10g', modes[0], 0.3]]

gr.Interface(
    fn=func,
    inputs=[
        gr.Image(label='Input', type='numpy'),
        gr.Radio(label='Model', choices=model_sizes, type='value',
                 value='10g'),
        gr.Radio(label='Mode', choices=modes, type='index', value=modes[0]),
        gr.Slider(label='Face Score Threshold',
                  minimum=0,
                  maximum=1,
                  step=0.05,
                  default=0.3),
    ],
    outputs=gr.Image(label='Output', type='numpy'),
    examples=examples,
    title=TITLE,
    description=DESCRIPTION,
).queue().launch(show_api=False)
