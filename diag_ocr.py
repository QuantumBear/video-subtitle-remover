# -*- coding: utf-8 -*-
"""OCR 检出行为诊断:对比服务器与本机对开头大字的检出差异。

用法: python diag_ocr.py <视频路径>
预期(本机基准): f20 裁剪送检 3 框 / f50 裁剪送检 2 框
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import av
from vsr_pipeline import Pipeline

import torch
print(f'torch {torch.__version__} | cuda {torch.cuda.is_available()}')
import paddle
print(f'paddle {paddle.__version__}')
import numpy
print(f'numpy {numpy.__version__}')
import cv2 as _cv2
print(f'opencv {_cv2.__version__}')


def get(video, idx):
    c = av.open(video)
    for i, f in enumerate(c.decode(video=0)):
        if i == idx:
            return np.asarray(f.to_image())


video = sys.argv[1] if len(sys.argv) > 1 else 'TikSave.io_7635080993354878239.mp4'
pipe = Pipeline()
region = (450, 1010, 0, 720)

for idx in (20, 50):
    img = get(video, idx)
    print(f'--- f{idx} | 全帧shape={img.shape} ---')
    # 裁剪送检(vsr_pipeline 当前实现)
    boxes = pipe.detect(img, region)
    print(f'裁剪送检: {len(boxes)}框 {boxes}')
    # 全帧送检对照
    ymin, ymax, xmin, xmax = region
    results = pipe.ocr.predict(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cnt = sum(len(r['dt_polys']) for r in results if r.get('dt_polys') is not None)
    print(f'全帧送检: {cnt}框(未过滤)')
