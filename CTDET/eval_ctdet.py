#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 15:03:54 2024

@author: arash
"""

from progress.bar import Bar
import time
from .src.lib.models.decode import ctdet_decode
from .src.lib.detectors.ctdet import CtdetDetector
from .src.lib.utils.utils import AverageMeter
import numpy as np
import openvino as ov
import torch


def evaluate_ctdet(data_loader, dataset, detector, opt):
    results = {}
    num_iters = len(dataset)
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
    avg_time_stats = {t: AverageMeter() for t in time_stats}
    for ind, (img_id, pre_processed_images) in enumerate(data_loader):
      ret = detector.run(pre_processed_images)
      results[img_id.numpy().astype(np.int32)[0]] = ret['results']
      Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                     ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
      for t in avg_time_stats:
        avg_time_stats[t].update(ret[t])
        Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
          t, tm = avg_time_stats[t])
      bar.next()
    bar.finish()
    dataset.run_eval(results, opt.save_dir)
    



def IR_infer(infer, images, opt, input_layer, return_time=True):
    images = images.numpy()
    min_val = images.min()
    max_val = images.max()
    
    images_norm = 256.0 * (images - min_val) / (max_val - min_val)
    
    infer.set_tensor(input_layer, ov.Tensor(images_norm))
    infer.start_async()
    res_ov = infer.get_output_tensor
    infer.wait()
    
    hm = torch.from_numpy(res_ov(0).data)
    wh = torch.from_numpy(res_ov(1).data)
    reg = torch.from_numpy(res_ov(2).data)
    output = {
        "hm": hm,
        "wh": wh,
        "reg": reg}
    
    
    hm = hm.sigmoid_()
    forward_time = time.time()
    dets = ctdet_decode(hm, wh, reg=reg, cat_spec_wh=opt.cat_spec_wh, K=opt.K)
    
    return output, dets, forward_time


def ov_inf_ret(ct, infer, pre_processed_images, opt, input_layer):
    
    load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
    merge_time, tot_time = 0, 0
    start_time = time.time()
    
    scale_start_time = time.time()
    
    images = pre_processed_images['images'][opt.test_scales[0]][0]
    meta = pre_processed_images['meta'][opt.test_scales[0]]
    meta = {k: v.numpy()[0] for k, v in meta.items()}
    
    pre_process_time = time.time()
    pre_time += pre_process_time - scale_start_time
    
    output, dets, forward_time = IR_infer(infer, images, opt, input_layer, return_time=True)
    
    net_time += forward_time - pre_process_time
    decode_time = time.time()
    dec_time += decode_time - forward_time
    
    dets = ct.post_process(dets, meta, opt.test_scales[0])
    
    post_process_time = time.time()
    post_time += post_process_time - decode_time
    
    detections = [dets]
    
    results = ct.merge_outputs(detections)
    end_time = time.time()
    merge_time += end_time - post_process_time
    tot_time += end_time - start_time
    
    return {'results': results, 'tot': tot_time, 'load': load_time,
            'pre': pre_time, 'net': net_time, 'dec': dec_time,
            'post': post_time, 'merge': merge_time}


def evaluate_ctdet_IR(data_loader, dataset, compiled_ov_model, opt):
    "should add ov model infer"
    results = {}
    num_iters = len(dataset)
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
    avg_time_stats = {t: AverageMeter() for t in time_stats}
    input_layer = compiled_ov_model.input(0)
    infer = compiled_ov_model.create_infer_request()
    ct = CtdetDetector(opt)
    for ind, (img_id, pre_processed_images) in enumerate(data_loader):
      ret = ov_inf_ret(ct, infer, pre_processed_images, opt, input_layer)
      results[img_id.numpy().astype(np.int32)[0]] = ret['results']
      Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                     ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
      for t in avg_time_stats:
        avg_time_stats[t].update(ret[t])
        Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
          t, tm = avg_time_stats[t])
      bar.next()
    bar.finish()
    dataset.run_eval(results, opt.save_dir)
