"""
This code is based on the official quantization flow introduced in:
    
    https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html
    
The IR model input for both original model and quantized model is:
    Numpy array of shape (1, 3, 512, 512) with values between 0.0 and 255.0 (only float32)
    
"""

import torch
from typing import Dict
import nncf
import openvino as ov
from pathlib import Path
import argparse
import openvino.properties.hint as hints
from CTDET.src.lib.datasets.dataset_factory import dataset_factory
from CTDET.src.lib.detectors.detector_factory import detector_factory
from CTDET.src.lib.opts import opts
from CTDET.src.test import PrefetchDataset
from CTDET import eval_ctdet as ev

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
        _, data_dict = data_item
        img = data_dict['images'][1.0][0]
        
        img = img.numpy()
        min_val = img.min()
        max_val = img.max()
        
        input_tensor = 256.0 * (img - min_val) / (max_val - min_val)
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
    opt = opts().init(['ctdet', '--arch', 'dlav0_34'])
    Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    opt.data_dir = '/home/arash/Thesis/2-quantization app/datasets/'
    opt.gpus = [-1]
    split = 'val' 
    dataset = Dataset(opt, split)
    # Adjust this value according to your .pth file
    opt.load_model = '/home/arash/Thesis/2-quantization app/CTDET/ctdet_coco_dlav0_1x.pth'
    # Provide this to store a .json report file
    opt.save_dir = '/home/arash/Thesis/2-quantization app/CTDET/exp_dir'
    Detector = detector_factory[opt.task]
    detector = Detector(opt)

    data_loader = torch.utils.data.DataLoader(
      PrefetchDataset(opt, dataset, detector.pre_process), 
      batch_size=1, shuffle=False, num_workers=4, pin_memory=False)
    
    

    if mode == "qnt":
        print("..............")
        print("Quantization is starting...")
        print("..............")
        
        ov_model_path = Path(f"{ROOT}/ctdet_openvino_model/ctdet_coco_dlav0_512.xml")
        ov_model = ov.Core().read_model(ov_model_path)
        ov_model.reshape({0: [1, 3, 512, 512]})  # to reduce inference complexity

        # Quantize model
        quantized_model = quantize(ov_model, data_loader)
        quantized_model_path = Path(f"{ROOT}/ctdet_openvino_model/ctdet_quantized.xml")
        ov.save_model(quantized_model, str(quantized_model_path))
        return
    
    else:
        core = ov.Core()
  
        if mode == "eval_orig":
            print("..............")
            print("Evaluation of the original IR model...")
            print("..............")
            ov_model_path = Path(f"{ROOT}/ctdet_openvino_model/ctdet_coco_dlav0_512.xml")
            ov_model = ov.Core().read_model(ov_model_path)
            ov_model.reshape({0: [1, 3, 512, 512]})
            
            compiled_model = core.compile_model(ov_model, device_name="CPU", config={hints.performance_mode: hints.PerformanceMode.THROUGHPUT})
            ev.evaluate_ctdet_IR(data_loader, dataset, compiled_model, opt)
            
            return
        else:
            print("..............")
            print("Evaluation of the quantized IR model...")
            print("..............")
            ov_model_path = Path(f"{ROOT}/ctdet_openvino_model/ctdet_quantized.xml")
            ov_model = ov.Core().read_model(ov_model_path)
            # Quantized model has the 512*512 shape for input by default
            #ov_model.reshape({0: [1, 3, 512, 512]})
            compiled_model = core.compile_model(ov_model, device_name="CPU", config={hints.performance_mode: hints.PerformanceMode.THROUGHPUT})
            ev.evaluate_ctdet_IR(data_loader, dataset, compiled_model, opt)
            return
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model selection")
    parser.add_argument(
        '--mode', 
        choices=['qnt', 'eval_orig', 'eval_qnt'],  # Define allowed options
        required=True,  # Make the mode argument mandatory
        help="Mode of operation: 'qnt' for quantization, 'eval_orig' for original model evaluation, 'eval_qnt' for quantized model evaluation"
    )
    marg = parser.parse_args()

    main(marg.mode)
