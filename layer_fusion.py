"""
Function for fusing convolutional layer with their following batch normalization
"""



def fuse_models(model_list_dir):
    """
    arges:
    model_list_dir (str): The directory of all models. Ex: ./mydevice/home/mymodels/
    """
  
    import os
    from ultralytics import YOLO
    for file in os.listdir(model_list_dir):
        model_path = os.path.join(model_list_dir, file)
        model = YOLO(model_path, task='detect')
        model.fuse()
        model.export(format='openvino', optimize = False, half = False, int8 = False)
        
        
