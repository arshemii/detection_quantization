"""
Post-training quantization with validation
This script is based on the main openvino script (A few adjustment):
Source: https://github.com/openvinotoolkit/nncf/blob/develop/examples/post_training_quantization/openvino/yolov8/main.py
"""


import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Tuple
import argparse
import numpy as np
import openvino as ov
import torch
from tqdm import tqdm
from ultralytics.cfg import get_cfg
from ultralytics.data.converter import coco80_to_coco91_class
from ultralytics.data.utils import check_det_dataset
from ultralytics.engine.validator import BaseValidator as Validator
from ultralytics.models.yolo import YOLO
from ultralytics.utils import DATASETS_DIR
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.metrics import ConfusionMatrix
import nncf

ROOT = Path(__file__).parent.resolve()

def validate(
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


def prepare_validation(model: YOLO, args: Any) -> Tuple[Validator, torch.utils.data.DataLoader]:
    validator = model.task_map[model.task]["validator"](args=args)
    validator.data = check_det_dataset(args.data)   # Check if the dataset is in the same directory as default data path in confihuration
    validator.stride = 32
    dataset = validator.data["val"]
    print(f"{dataset}")

    data_loader = validator.get_dataloader(f"{DATASETS_DIR}/coco", 1)

    validator.is_coco = True
    validator.class_map = coco80_to_coco91_class()
    validator.names = model.model.names
    validator.metrics.names = validator.names
    validator.nc = model.model.model[-1].nc

    return validator, data_loader


def benchmark_performance(model_path, config) -> float:
    command = f"benchmark_app -m {model_path} -d CPU -api async -t 30"
    command += f' -shape "[1,3,{config.imgsz},{config.imgsz}]"'
    cmd_output = subprocess.check_output(command, shell=True)  # nosec

    match = re.search(r"Throughput\: (.+?) FPS", str(cmd_output))
    return float(match.group(1))


def prepare_openvino_model(model: YOLO, model_name: str) -> Tuple[ov.Model, Path]:
    ir_model_path = Path(f"{ROOT}/{model_name}_openvino_model/{model_name}.xml")  # .xml shall be in a direcory with this format: {model_name}_openvino_model
    return ov.Core().read_model(ir_model_path), ir_model_path


def quantize(model: ov.Model, data_loader: torch.utils.data.DataLoader, validator: Validator) -> ov.Model:
    def transform_fn(data_item: Dict):
        """
        Quantization transform function. Extracts and preprocess input data from dataloader
        item for quantization.
        Parameters:
        data_item: Dict with data item produced by DataLoader during iteration
        Returns:
            input_tensor: Input data for quantization
        """
        input_tensor = validator.preprocess(data_item)["img"].numpy()
        return input_tensor

    quantization_dataset = nncf.Dataset(data_loader, transform_fn)


    quantized_model = nncf.quantize(
        model,
        quantization_dataset,
        #mode = "fp8_e4m3",   # Not available
        subset_size=len(data_loader),
        preset=nncf.QuantizationPreset.PERFORMANCE,  # You can change to ACCURACY if throughput is less important
        target_device = nncf.TargetDevice.CPU,       # CPU for LattePanda
        fast_bias_correction=True,
        ignored_scope=nncf.IgnoredScope(
            types=["Multiply", "Subtract", "Sigmoid"],
            subgraphs=[
              # Use netron if you have a different model
              # See the comment after this function if you are optimizing YOLOv8
                nncf.Subgraph(
                    inputs=['__module.model.23/aten::cat/Concat',
                              '__module.model.23/aten::cat/Concat_1',
                              '__module.model.23/aten::cat/Concat_2'],
                      outputs=['__module.model.23/aten::cat/Concat_3'])
            ],
        ),
    )
    return quantized_model
  
             # YOLOv8 variations have the following subgraph:
             #     inputs=['__module.model.22/aten::cat/Concat',
             #               '__module.model.22/aten::cat/Concat_1',
             #               '__module.model.22/aten::cat/Concat_2'],
             #       outputs=['__module.model.22/aten::cat/Concat_3']


def main(model_name):
    model = YOLO(f"{ROOT}/model_w_orig/{marg.model_name}.pt")  # Use the same directories as this repository
    args = get_cfg(cfg=DEFAULT_CFG)
    args.data = "coco.yaml"
    # Prepare validation dataset and helper
    validator, data_loader = prepare_validation(model, args)
    # Convert to OpenVINO model
    ov_model, ov_model_path = prepare_openvino_model(model, marg.model_name)
    # Quantize mode in OpenVINO representation
    quantized_model = quantize(ov_model, data_loader, validator)
    quantized_model_path = Path(f"{ROOT}/{marg.model_name}_openvino_model/{marg.model_name}_quantized.xml") # It stores new xml and bin inside the same
                                                                                                            # directory as non-quantized xml and bin
    ov.save_model(quantized_model, str(quantized_model_path))

    # Validate FP32 model
    fp_stats, total_images, total_objects = validate(ov_model, tqdm(data_loader), validator)
    print("Floating-point model validation results:")
    print_statistics(fp_stats, total_images, total_objects)

    # Validate quantized model
    q_stats, total_images, total_objects = validate(quantized_model, tqdm(data_loader), validator)
    print("Quantized model validation results:")
    print_statistics(q_stats, total_images, total_objects)

    Benchmark performance of FP32 model
    fp_model_perf = benchmark_performance(ov_model_path, args)
    print(f"Floating-point model performance: {fp_model_perf} FPS")

    Benchmark performance of quantized model
    quantized_model_perf = benchmark_performance(quantized_model_path, args)
    print(f"Quantized model performance: {quantized_model_perf} FPS")

    return fp_stats["metrics/mAP50-95(B)"], q_stats["metrics/mAP50-95(B)"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model validation and performance benchmark.")
    parser.add_argument("model_name", type=str, help="Name of the model (e.g., yolov8n)")
    marg = parser.parse_args()

    main(marg.model_name) # example use: python3 quantization.py yolov8
