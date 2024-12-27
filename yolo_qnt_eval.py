"""
This code is based on the official quantization flow introduced in:
    
    https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html
    

    
"""



from pathlib import Path
import argparse
import openvino as ov
import torch
from tqdm import tqdm
from ultralytics.cfg import get_cfg
from ultralytics.engine.validator import BaseValidator as Validator
from ultralytics.models.yolo import YOLO
from ultralytics.utils import DEFAULT_CFG
import nncf
import YOLO_u.yolo_utils as ut
from typing import Dict


def quantize(model: ov.Model, data_loader: torch.utils.data.DataLoader, validator: Validator, ignored) -> ov.Model:
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
        #mode = "fp8_e4m3",
        subset_size=len(data_loader),
        preset=nncf.QuantizationPreset.PERFORMANCE,
        target_device = nncf.TargetDevice.CPU,
        fast_bias_correction=True,
        ignored_scope=ignored
    )
    return quantized_model



def main(arg):
    
    ROOT = Path.cwd().resolve()
    
    data_dir = Path(f"{ROOT}/datasets")
    
    if arg.m =='8':
        model_name = 'yolov8n'
        yolo_pt_path = Path('YOLO_u/yolov8n.pt')
        ignored =nncf.IgnoredScope(
            types=["Multiply", "Subtract", "Sigmoid"],
            subgraphs=[
                nncf.Subgraph(
                    inputs=['__module.model.22/aten::cat/Concat',
                              '__module.model.22/aten::cat/Concat_1',
                              '__module.model.22/aten::cat/Concat_2'],
                      outputs=['__module.model.22/aten::cat/Concat_3'])
            ],
        )
    elif arg.m == '11':
        model_name = 'yolo11n'
        yolo_pt_path = Path('YOLO_u/yolo11n.pt')
        ignored =nncf.IgnoredScope(
            types=["Multiply", "Subtract", "Sigmoid"],
            subgraphs=[
                nncf.Subgraph(
                    inputs=['__module.model.23/aten::cat/Concat',
                              '__module.model.23/aten::cat/Concat_1',
                              '__module.model.23/aten::cat/Concat_2'],
                      outputs=['__module.model.23/aten::cat/Concat_3'])
            ],
        )
    else:
        model_name = 'yolov5nu'
        yolo_pt_path = Path('YOLO_u/yolov5nu.pt')
        ignored = None
    
    pt_model = YOLO(yolo_pt_path)
    model_args = get_cfg(cfg=DEFAULT_CFG)
    model_args.data = "coco.yaml"
    
    validator, data_loader = ut.prepare_validation(pt_model, model_args, data_dir)
    
    if arg.mode == 'eval_qnt':
        ov_model, ov_model_path = ut.prepare_openvino_model(model_name, ROOT, arg)
        
        fp_stats, total_images, total_objects = ut.validate_ov(ov_model, tqdm(data_loader), validator)
        print("Quantized model validation results:")
        ut.print_statistics(fp_stats, total_images, total_objects)
        return fp_stats["metrics/mAP50-95(B)"], fp_stats["metrics/mAP50(B)"]
        
    else:
        ov_model, ov_model_path = ut.prepare_openvino_model(model_name, ROOT, arg)
        if arg.mode == 'qnt':
            quantized_model = quantize(ov_model, data_loader, validator, ignored)
            quantized_model_path = Path(f"{ROOT}/{model_name}_openvino_model/{model_name}_quantized.xml")
            ov.save_model(quantized_model, str(quantized_model_path))
            print("Quantization is finished")
        else:
            q_stats, total_images, total_objects = ut.validate_ov(ov_model, tqdm(data_loader), validator)
            print("Original model validation results:")
            ut.print_statistics(q_stats, total_images, total_objects)
            return q_stats["metrics/mAP50-95(B)"], q_stats["metrics/mAP50(B)"]



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model selection")
    parser.add_argument(
        '--mode', 
        choices=['qnt', 'eval_orig', 'eval_qnt'],  # Define allowed options
        required=True,  # Make the mode argument mandatory
        help="Mode of operation: 'qnt' for quantization, 'eval_orig' for original model evaluation, 'eval_qnt' for quantized model evaluation"
    )
    parser.add_argument(
        '--m', 
        choices=['8', '11', '5'],  # Define allowed options
        required=True,  # Make the mode argument mandatory
        help="YOLO variation selection: 8 for yolov8n and 11 for yolov11n and 5 for yolov5nu"
    )
    marg = parser.parse_args()

    main(marg)
