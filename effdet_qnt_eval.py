"""
This code is based on the official quantization flow introduced in:
    
    https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html
    
The IR model input for both original model and quantized model is:
    Numpy array of shape (3, 512, 512, 3) with values between 0.0 and 255.0 (only float32)
    
"""



from typing import Dict
import nncf
import openvino as ov
from pathlib import Path
import argparse
from EFFDET import ir_eval as ut
import tensorflow as tf

ROOT = Path.cwd().resolve()

def quantize(model: ov.Model, data_loader) -> ov.Model:
    def transform_fn(data_item: Dict):
        """
        Quantization transform function. Extracts and preprocess input data from dataloader
        item for quantization.
        Parameters:
        data_item: Dict with data item produced by DataLoader during iteration
        Returns:
            input_tensor: Input data for quantization
        """
        input_tensor, _, _ = ut.img_prepare(data_item, './datasets/coco/images/val2017')
        return input_tensor

    quantization_dataset = nncf.Dataset(data_loader, transform_fn)

    quantized_model = nncf.quantize(
        model,
        quantization_dataset,
        #mode = "fp8_e4m3",
        subset_size=5000,
        preset=nncf.QuantizationPreset.PERFORMANCE,
        target_device = nncf.TargetDevice.CPU,
        fast_bias_correction=True)
    
    return quantized_model


def main(mode):

    coco_ann_path = './datasets/coco/annotations/instances_val2017.json'
    tfrec_path = './EFFDET/tfrecord/val-00000-of-00001.tfrecord'
    data_path = './datasets/coco/images/val2017'
    
    FLAGS = ut.args_maker(tfrec_path, coco_ann_path, data_path)
    
    ds = ut.DatasetWithLength(FLAGS.val_file_pattern, 5000)

    if mode == "qnt":
        print("..............")
        print("Quantization is starting...")
        print("..............")
        
        ov_model_path = Path(f"{ROOT}/effdet_openvino_model/efficientdet-d0-tf.xml")
        ov_model = ov.Core().read_model(ov_model_path)

        # Quantize model
        quantized_model = quantize(ov_model, ds)
        quantized_model_path = Path(f"{ROOT}/effdet_openvino_model/effdet_quantized.xml")
        ov.save_model(quantized_model, str(quantized_model_path))
        return
    
    else:
        if mode == "eval_orig":
            print("..............")
            print("Evaluation of the original IR model...")
            print("..............")
            ov_model_path = Path(f"{ROOT}/effdet_openvino_model/efficientdet-d0-tf.xml")
            infer, input_layer = ut.infer_prepare(ov_model_path, FLAGS)
            ut.evaluation_IR(FLAGS, infer, input_layer, ds)
            return
        
        else:
            print("..............")
            print("Evaluation of the quantized IR model...")
            print("..............")
            ov_model_path = Path(f"{ROOT}/effdet_openvino_model/effdet_quantized.xml")
            infer, input_layer = ut.infer_prepare(ov_model_path, FLAGS)
            ut.evaluation_IR(FLAGS, infer, input_layer, ds)
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
    
    
    
