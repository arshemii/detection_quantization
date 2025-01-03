"""
This code is based on the official quantization flow introduced in:
    
    https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html
    
The IR model input for both original model and quantized model is:
    Numpy array of shape (1, 3, 600, 800) with values between 0.0 and 255.0 (only float32)
    
"""

from torch.utils.data import DataLoader
import torch
from typing import Dict
import nncf
import openvino as ov
from pathlib import Path
import torch.nn.functional as F
import argparse
import openvino.properties.hint as hints
from DETR.eval_ir import evaluate_ov
from DETR.models.detr import SetCriterion, PostProcess
from DETR.models.matcher import build_matcher
from DETR.datasets import get_coco_api_from_dataset
import DETR.util.misc as utils
from DETR.datasets import build_dataset



ROOT = Path.cwd().resolve()


def quantize(model: ov.Model, data_loader: torch.utils.data.DataLoader) -> ov.Model:
    def transform_fn(data_item: Dict):
        """
        Quantization transform function. Extracts and preprocess input data from dataloader
        item for quantization.
        Parameters:
        data_item: Dict with data item produced by DataLoader during iteration
        Returns:
            input_tensor: Input data for quantization
        """
        img, target = data_item
        samples = img.tensors
        
        samples = F.interpolate(samples, size=(600, 800), mode='bilinear', align_corners=False)
        samples = samples.numpy()
        samples = samples[:, :, :, ::-1]
        array_min = samples.min()
        array_max = samples.max()

        # Normalize the array to the range [0, 1]
        samples = (samples - array_min) / (array_max - array_min)
        input_tensor = samples*255.0
        return input_tensor

    quantization_dataset = nncf.Dataset(data_loader, transform_fn)

    quantized_model = nncf.quantize(
        model,
        quantization_dataset,
        #mode = "fp8_e4m3",
        subset_size=len(data_loader),
        preset=nncf.QuantizationPreset.PERFORMANCE,
        target_device = nncf.TargetDevice.CPU,
        fast_bias_correction=True)
    
    return quantized_model


def main(mode):
    class Args:
        def __init__(self):
            self.dataset_file = 'coco'
            # Adjust the coco_path according to your path
            self.coco_path = Path(f"{ROOT}/datasets/coco")
            self.batch_size = 1
            self.num_workers = 4
            self.masks = False
            self.bbox_loss_coef = 5
            self.giou_loss_coef = 2
            self.output_dir = Path(f"{ROOT}/eval_out/")
            self.device = 'cpu'
            self.set_cost_class = 1
            self.set_cost_bbox = 5
            self.set_cost_giou = 2
            self.eos_coef = 0.1
            self.num_classes = 91
            
            
    args = Args()
    
    # Creating data loader
    dataset_val = build_dataset(image_set='val', args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if mode == "qnt":
        print("..............")
        print("Quantization is starting...")
        print("..............")
        
        ov_model_path = Path(f"{ROOT}/models/detr_openvino_model/detr-resnet50.xml")
        ov_model = ov.Core().read_model(ov_model_path)
        ov_model.reshape({0: [1, 3, 600, 800]})  # to reduce inference complexity

        # Quantize model
        quantized_model = quantize(ov_model, data_loader_val)
        quantized_model_path = Path(f"{ROOT}/models/detr_openvino_model/detr_quantized.xml")
        ov.save_model(quantized_model, str(quantized_model_path))
        return
    
    else:
        matcher = build_matcher(args)
        losses = ['labels', 'boxes', 'cardinality']
        weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
        weight_dict['loss_giou'] = args.giou_loss_coef

        criterion = SetCriterion(args.num_classes, matcher=matcher, weight_dict=weight_dict,
                                 eos_coef=args.eos_coef, losses=losses)

        postprocessors = {'bbox': PostProcess()}
        
        base_ds = get_coco_api_from_dataset(dataset_val)
        
        core = ov.Core()
        device = torch.device(args.device)
        
        if mode == "eval_orig":
            print("..............")
            print("Evaluation of the original IR model...")
            print("..............")
            ov_model_path = Path(f"{ROOT}/detr_openvino_model/detr-resnet50.xml")
            ov_model = ov.Core().read_model(ov_model_path)
            ov_model.reshape({0: [1, 3, 600, 800]})
            compiled_model = core.compile_model(ov_model, device_name="CPU", config={hints.performance_mode: hints.PerformanceMode.THROUGHPUT})
            input_layer = compiled_model.input(0)
            test_stats_main, coco_evaluator_main = evaluate_ov(compiled_model, criterion, postprocessors,
                                                            data_loader_val, base_ds, device, args.output_dir, input_layer)
            return
        else:
            print("..............")
            print("Evaluation of the quantized IR model...")
            print("..............")
            ov_model_path = Path(f"{ROOT}/models/detr_openvino_model/detr_quantized.xml")
            ov_model = ov.Core().read_model(ov_model_path)
            # Quantized model has the 600*800 shape for input by default
            #ov_model.reshape({0: [1, 3, 600, 800]})
            compiled_model = core.compile_model(ov_model, device_name="CPU", config={hints.performance_mode: hints.PerformanceMode.THROUGHPUT})
            input_layer = compiled_model.input(0)
            test_stats_main, coco_evaluator_main = evaluate_ov(compiled_model, criterion, postprocessors,
                                                            data_loader_val, base_ds, device, args.output_dir, input_layer)
            return
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model selection")
    parser.add_argument(
        '--mode', 
        choices=['qnt', 'eval_orig', 'eval_qnt'],
        required=True,
        help="Mode of operation: 'qnt' for quantization, 'eval_orig' for original model evaluation, 'eval_qnt' for quantized model evaluation"
    )
    marg = parser.parse_args()

    main(marg.mode)
