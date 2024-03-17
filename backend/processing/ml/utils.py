import os
from pathlib import Path

import torch
from ultralytics import YOLO

from .segmentation.SegNet import SegNet


def get_yolo_model(device):
    module_path = Path(__file__).resolve().parent
    if os.path.exists(module_path / './yolov8m-pose.pt'):
        return YOLO(module_path / './yolov8m-pose.pt')

    return YOLO('yolov8m-pose.pt').to(device)


def get_seg_model(device):
    seg_model = SegNet().to(device)
    module_path = Path(__file__).resolve().parent
    seg_model.load_state_dict(
        torch.load(module_path / './segnet_bce_1125_45_epoch.pth', map_location=torch.device(device)))
