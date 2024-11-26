# yolo_optimization
A simple script for optimizing YOLO models (v8 and v11)


--------------------------------
This repository is nothing but a simple script for optimizing YOLOv8 and YOLOv11 models.

It contains:
1. A simple layer fusing to have less complexity by fusing convolution layers and batch normalization layers.
   To learn about layer fusing:          https://proceedings.mlr.press/v157/o-neill21a/o-neill21a.pdf
2. A quantization workflow using "nncf" toolkit of OpenVino.
   To learn more about quantization:     https://paperswithcode.com/paper/post-training-quantization-for-neural
   To learn more about nncf:             https://docs.openvino.ai/2024/openvino-workflow/model-optimization.html
--------------------------------
Github resources:
1. https://github.com/ultralytics/ultralytics
2. https://github.com/openvinotoolkit/nncf
--------------------------------
You can find 4 optimized model in:
   https://github.com/arshemii/drone_od_infer/tree/main/models

--------------------------------
# Contact: arshemii1373@gmail.com
