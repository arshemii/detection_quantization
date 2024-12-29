import argparse
import openvino as ov
import openvino.properties.hint as hints
import tensorflow as tf
import os
import cv2
import numpy as np
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import time


def args_maker(tfrecord_path, coco_ann_path, data_dir):

    FLAGS = argparse.Namespace()
    FLAGS.batch_size = 1
    FLAGS.val_file_pattern = tfrecord_path
    FLAGS.eval_samples = 5000
    FLAGS.val_json_file = coco_ann_path
    FLAGS.ov_config = hints.PerformanceMode.THROUGHPUT
    FLAGS.device = "CPU"
    FLAGS.data_dir = data_dir
    return FLAGS


def infer_prepare(model_path, FLAGS):
    core = ov.Core()
    model = core.read_model(model_path)
    compiled_model = core.compile_model(model,
                                        device_name=FLAGS.device,
                                        config={hints.performance_mode: FLAGS.ov_config})
    infer = compiled_model.create_infer_request()
    input_layer = compiled_model.input(0)
    return infer, input_layer

def img_prepare(image, data_dir):
    parsed = tf.train.Example.FromString(image.numpy())
    path = parsed.features.feature['image/filename']
    img_id = parsed.features.feature['image/source_id']
    img_name = path.bytes_list.value[0].decode('utf-8')
    img_id = img_id.bytes_list.value[0].decode('utf-8')
    directory = data_dir
    img_path = os.path.join(directory, img_name)
    image = cv2.imread(img_path)  # Read as BGR format
    image_resized = cv2.resize(image, (512, 512))
    image_tensor = np.expand_dims(image_resized, axis=0)
    image_tensor = image_tensor.astype(np.float32)
    scale = (image.shape[0], image.shape[1])
    return image_tensor, img_id, scale

def det_pp_coco(detections, img_id, results, scale):
    scale_h = scale[0]
    scale_w = scale[1]
    detections[:, :, 5] = detections[:, :, 5] - detections[:, :, 3]  # w (x2 - x1)
    detections[:, :, 6] = detections[:, :, 6] - detections[:, :, 4]  # h (y2 - y1)
    # detections is 
    for bbox in detections[0]:
        detection = {
            'image_id': int(img_id),
            'category_id': int(bbox[1])+1,
            'bbox': [float(scale_w*bbox[3]), float(scale_h*bbox[4]), float(scale_w*bbox[5]), float(scale_h*bbox[6])],
            'score': float(bbox[2])}
        results.append(detection)
    return results


def evaluator(results, ann_path):
    annType = 'bbox'
    
    res_file_name = 'results.json'
    with open(res_file_name, 'w') as f:
        json.dump(results, f, indent=4) 
        
    cocoGt=COCO(ann_path)
    cocoDt=cocoGt.loadRes(res_file_name)
    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    os.remove(res_file_name)
    
    
    
    
def evaluation_IR(FLAGS, infer, input_layer, ds):
        
    pbar = tf.keras.utils.Progbar((FLAGS.eval_samples + FLAGS.batch_size - 1) // FLAGS.batch_size)
    
    results = []
    i=0 
    time.sleep(2)
    print("Running Prediction on dataset ...")
    t1 = time.time()
    for image in ds:
        image, img_id, scale = img_prepare(image, FLAGS.data_dir)
        infer.set_tensor(input_layer, ov.Tensor(image))
        infer.start_async()
        infer.wait()
        res = infer.get_output_tensor(0).data[0]
        
        results = det_pp_coco(res, img_id, results, scale)
        
        pbar.update(i)
        i += 1

    total_time = int(time.time() - t1)

    print("Prediction is done, preparing for evaluation ...")
    time.sleep(2)
    evaluator(results, FLAGS.val_json_file)
    print(f"Evaluation is done and prediction took {total_time} seconds!")


class DatasetWithLength:
    def __init__(self, val_file_pattern, length):
        self.dataset = tf.data.TFRecordDataset(val_file_pattern)
        self.len = length

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return self.len

    def __getattr__(self, attr):
        return getattr(self.dataset, attr)
    
    

