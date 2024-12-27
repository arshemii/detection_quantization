#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 16:34:49 2024

@author: arash
"""

# Pre and post process from .models/yolo.detect/val.py



from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import openvino as ov
import torch
from ultralytics.data.converter import coco80_to_coco91_class
from ultralytics.data.utils import check_det_dataset
from ultralytics.engine.validator import BaseValidator as Validator
from ultralytics.models.yolo import YOLO
from ultralytics.utils.metrics import ConfusionMatrix


def prepare_validation(model: YOLO, args: Any, data_dir) -> Tuple[Validator, torch.utils.data.DataLoader]:
    validator = model.task_map[model.task]["validator"](args=args)
    validator.data = check_det_dataset(args.data)
    validator.stride = 32
    dataset = validator.data["val"]
    print(f"{dataset}")

    data_loader = validator.get_dataloader(f"{data_dir}/coco", 1)

    validator.is_coco = True
    validator.class_map = coco80_to_coco91_class()
    validator.names = model.model.names
    validator.metrics.names = validator.names
    validator.nc = model.model.model[-1].nc

    return validator, data_loader

def validate_ov(
    model: ov.Model, data_loader: torch.utils.data.DataLoader, validator: Validator, num_samples: int = None
) -> Tuple[Dict, int, int]:
    validator.seen = 0
    validator.jdict = []
    validator.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
    validator.confusion_matrix = ConfusionMatrix(nc=validator.nc)
    model.reshape({0: [1, 3, -1, -1]})
    compiled_model = ov.compile_model(model, device_name="CPU")
    output_layer = compiled_model.output(0)
    for batch_i, batch in enumerate(data_loader):
        if num_samples is not None and batch_i == num_samples:
            break
        batch = validator.preprocess(batch)
        preds = torch.from_numpy(compiled_model(batch["img"])[output_layer])
        preds = validator.postprocess(preds)
        validator.update_metrics(preds, batch)
    stats = validator.get_stats()
    return stats, validator.seen, validator.nt_per_class.sum()


def prepare_openvino_model(model_name: str, ROOT, arg) -> Tuple[ov.Model, Path]:
    if arg.mode == 'eval_orig' or arg.mode == 'qnt':
        ir_model_path = Path(f"{ROOT}/{model_name}_openvino_model/{model_name}.xml")
    else:
        ir_model_path = Path(f"{ROOT}/{model_name}_openvino_model/{model_name}_quantized.xml")

    return ov.Core().read_model(ir_model_path), ir_model_path


def print_statistics(stats: np.ndarray, total_images: int, total_objects: int) -> None:
    mp, mr, map50, mean_ap = (
        stats["metrics/precision(B)"],
        stats["metrics/recall(B)"],
        stats["metrics/mAP50(B)"],
        stats["metrics/mAP50-95(B)"],
    )
    s = ("%20s" + "%12s" * 6) % ("Class", "Images", "Labels", "Precision", "Recall", "mAP@.5", "mAP@.5:.95")
    print(s)
    pf = "%20s" + "%12i" * 2 + "%12.3g" * 4  # print format
    print(pf % ("all", total_images, total_objects, mp, mr, map50, mean_ap))