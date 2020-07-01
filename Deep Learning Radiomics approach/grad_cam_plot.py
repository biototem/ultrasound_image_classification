# -*- coding: utf-8 -*-
# @Licence : bio-totem
"""
Created on Mon Aug 29 09:39:07 2018

@author: Kwong.Bohrium
"""


import os
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 10, 10
#import skimage.io as io
from keras.models import load_model
import keras.backend as K
import numpy as np
import cv2
import configparser


def grad_cam(model, x, category_index, layer_name):
    """
    Args:
       model: model
       x: image input
       category_index: category index
       layer_name: last convolution layer name
    """
    # get category loss
    class_output = model.output[:, category_index]
    
    # layer output
    convolution_output = model.get_layer(layer_name).output
    # get gradients
    grads = K.gradients(class_output, convolution_output)[0]
    # get convolution output and gradients for input
    gradient_function = K.function([model.input], [convolution_output, grads])
    
    output, grads_val = gradient_function([x])
    output, grads_val = output[0], grads_val[0]
    
    # avg
    if np.sum(np.unique(grads_val) == np.array([0])) != 1:
        weights = np.mean(grads_val, axis=2)
        cam = np.dot(np.mean(output,axis=2), weights.T)
    else:
        cam = np.mean(output,axis=2) 
    # create heat map
    cam = cv2.resize(cam, (x.shape[2], x.shape[1]), cv2.INTER_CUBIC)
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)
    
    # Return to BGR [0..255] from the preprocessed image
    image_rgb = x[0, :]
    image_rgb -= np.min(image_rgb)
    image_rgb = np.minimum(image_rgb, 255)
    
    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_COOL)
    cam = np.float32(cam) + np.float32(image_rgb)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap


if __name__ == "__main__":
    conf = configparser.ConfigParser()
    current_path = os.path.dirname(__file__)
    conf.read(os.path.join(current_path,"sys.ini"))    
    
    GPU_USE = conf.get("DEFAULT", "GPU_USE")
    #DEVICE = '\"/gpu:' + GPU_USE + '\"'
    print('\"/gpu:' + GPU_USE + '\"')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(GPU_USE))
    print('\"CUDA_VISIBLE_DEVICES\"'+ ' = ' '\"'+ GPU_USE + '\"')
    
    model1_path = conf.get("DEFAULT", "ELASTIC_CHECKPOINT_PATH")
    file_name = conf.get("DEFAULT", "ELASTIC_MODEL_NAME")
    model1 = load_model(os.path.join(model1_path,file_name))
    model2_path = conf.get("DEFAULT", "ULTRASONIC_CHECKPOINT_PATH")
    file_name = conf.get("DEFAULT", "ULTRASONIC_MODEL_NAME")
    model2 = load_model(os.path.join(model2_path,file_name))
    model_dict = {'elastic':model1,'ultrasonic':model2}
    #model_dict = {'elastic_Ultrasonic':model1,'Ultrasonic':model2}
    layer_dict = {'elastic':'max_pooling2d_4','ultrasonic':'max_pooling2d_2'}
    label_dict = {'benign':0,'malignant':1}
    dir_sub = conf.get("DEFAULT", "IMG_SOURCE_DIR")
    save_root_dir = conf.get("DEFAULT", "CAM_SAVE_ROOT_DIR")

    for root ,_, files in os.walk(dir_sub):
        for file in files:
            ori_img = cv2.imread(os.path.join(root,file))
            label = os.path.split(root)[-1].split('_')[0]
    
    #        print(os.path.split(root)[-1].split('_')[0])
            model = os.path.split(root)[-1].split('_')[1]
    #        print(os.path.split(root)[-1].split('_')[1])
    #        break
            cam, _ = grad_cam(model_dict[model], np.expand_dims(ori_img, axis=0),\
                              label_dict[label], layer_dict[model])
            save_dir = os.path.join(save_root_dir,os.path.split(root)[-1])
            if not os.path.isdir(save_dir): os.makedirs(save_dir)
            cv2.imwrite(os.path.join(save_dir,file),cam)
            print(os.path.join(os.path.split(root)[-1],file))