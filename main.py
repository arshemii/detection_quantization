#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 21:25:05 2024

@author: arash
"""
import argparse
import sys


def main(arg):
    
    
    
    script_args = ['--mode', arg.mode, '--arch', arg.arch]
    
    
    if arg.arch == 'ctdet':
        sys.argv = ['simpleqtz_ctdet.py'] + script_args
        script_path = 'simpleqtz_ctdet.py'
        with open(script_path, 'r') as script_file:
            script_code = script_file.read()
        exec(script_code, {"__name__": "__main__"})
    elif arg.arch == 'detr':
        sys.argv = ['simpleqtz_detr.py'] + script_args
        script_path = 'simpleqtz_detr.py'
        with open(script_path, 'r') as script_file:
            script_code = script_file.read()
        exec(script_code, {"__name__": "__main__"})
    else:
        script_args[2] = '--m'
        if arg.arch == 'yolo5':
            script_args[3] = '5'
        if arg.arch == 'yolo8':
            script_args[3] = '8'
        else:
            script_args[3] = '11'
        
        sys.argv = ['simpleqtz_yolo.py'] + script_args
        script_path = 'simpleqtz_yolo.py'
        with open(script_path, 'r') as script_file:
            script_code = script_file.read()
        exec(script_code, {"__name__": "__main__"})
    
 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model selection")
    parser.add_argument(
        '--mode', 
        choices=['qnt', 'eval_orig', 'eval_qnt'],  # Define allowed options
        required=True,  # Make the mode argument mandatory
        help="Mode of operation: 'qnt' for quantization, 'eval_orig' for original model evaluation, 'eval_qnt' for quantized model evaluation"
    )
    parser.add_argument(
        '--arch', 
        choices=['ctdet', 'detr', 'yolo5', 'yolo8', 'yolo11'],  # Define allowed options
        required=True,  # Make the mode argument mandatory
        help="Defining the model"
    )
    marg = parser.parse_args()

    main(marg)