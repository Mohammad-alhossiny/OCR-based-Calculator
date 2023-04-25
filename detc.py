import argparse
import time
from pathlib import Path
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel



class JersyDetector:
    def __init__(self, mdl_ctx) -> None:
        self._weights = mdl_ctx['weights']
        self._img_size = mdl_ctx['img-size']
        self._conf_thres = mdl_ctx["conf-thres"]
        self._iou_thres = mdl_ctx["iou-thres"]
        self._device = mdl_ctx["device"]
        self._agnostic_nms = mdl_ctx['agnostic-nms']
        self._augment = mdl_ctx['augment']
        self._no_track = mdl_ctx['no-track']
        self._trace = mdl_ctx['no-trace']
        self._classes = mdl_ctx['classes']

        # TODO: load the model
        self._device = select_device(self._device)
        self._model = attempt_load(
            self._weights, map_location=self._device)  # load FP32 model
        # half precision only supported on CUDA
        self._half = self._device.type != 'cpu'
        self._stride = int(self._model.stride.max())  # model stride
        self._imgsz = check_img_size(
            self._img_size, s=self._stride)  # check img_size

        if self._trace:
            self._model = TracedModel(
                self._model, self._device, self._img_size)

        if self._half:
            self._model.half()  # to FP16
        # Get names and colors
        self._names = self._model.module.names if hasattr(
            self._model, 'module') else self._model.names
        self._colors = [[random.randint(0, 255)
                         for _ in range(3)] for _ in self._names]

        # Run inference
        if self._device.type != 'cpu':
            self._model(torch.zeros(1, 3, self._imgsz, self._imgsz).to(
                self._device).type_as(next(self._model.parameters())))  # run once
        self._old_img_w = self._old_img_h = self._imgsz
        self._old_img_b = 1
    def detect(self, img: np.ndarray):
        save_img = False
        img = torch.from_numpy(img).to(self._device)
        img = img.half() if self._half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        img = img.permute((2, 0, 1))
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if self._device.type != 'cpu' and (self._old_img_b != img.shape[0] or self._old_img_h != img.shape[2] or self._old_img_w != img.shape[3]):
            self._old_img_b = img.shape[0]
            self._old_img_h = img.shape[2]
            self._old_img_w = img.shape[3]
            for i in range(3):
                self._model(img, augment=self._augment)[0]

        # Inference
        pred = self._model(img, augment=self._augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, self._conf_thres, self._iou_thres, classes=self._classes, agnostic=self._agnostic_nms)

        return pred