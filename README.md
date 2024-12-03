# Detection model quantization
A simple script for optimizing detection models on COCO using OpenVino Neural Network Comression Framework (NNCF)


--------------------------------
This repository is nothing but a simple script for optimizing detection models.

It contains:
1. A simple layer fusing to have less complexity by fusing convolution layers and batch normalization layers.
   To learn about layer fusing:          https://proceedings.mlr.press/v157/o-neill21a/o-neill21a.pdf
   ** No need for layer fusion if using OpenVino inference since the fusion is automatically done!
   
2. A quantization workflow using "nncf" toolkit of OpenVino.
   To learn more about quantization:     https://paperswithcode.com/paper/post-training-quantization-for-neural
   To learn more about nncf:             https://docs.openvino.ai/2024/openvino-workflow/model-optimization.html
   ** To use this function, you need a subset of coco dataset (or any other dataset in the format compatible with the detection models)
   ** See the comments in ModelName_quantization.py to adjust the code for your models
--------------------------------
Github resources:
1. https://github.com/ultralytics/ultralytics
2. https://github.com/openvinotoolkit/nncf

--------------------------------
1. YOLO_quantization.py for quantizing YOLO v8 and v11 (Small and nano)
2. CTDET_quantization.py for quantizing CenterNet dval0
--------------------------------
You can find some optimized models in:
   https://github.com/arshemii/drone_od_infer/tree/main/models

--------------------------------
# Contact: arshemii1373@gmail.com
