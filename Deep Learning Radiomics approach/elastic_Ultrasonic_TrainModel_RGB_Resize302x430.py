# -*- coding: utf-8 -*-
# @Licence : bio-totem
"""
Created on Mon Aug 27 09:39:07 2018

@author: zengyouling
"""
import os
import numpy as np
from skimage import io
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from sklearn.cross_validation import train_test_split

import configparser

from util_script import get_callbacks,get_model

if __name__ == "__main__":
    conf = configparser.ConfigParser()
    current_path = os.path.dirname(__file__)
    conf.read(os.path.join(current_path,"sys.ini"))    
    
    GPU_USE = conf.get("DEFAULT", "GPU_USE")
    #DEVICE = '\"/gpu:' + GPU_USE + '\"'
    print('\"/gpu:' + GPU_USE + '\"')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(GPU_USE))
    print('\"CUDA_VISIBLE_DEVICES\"'+ ' = ' '\"'+ GPU_USE + '\"')
    
    label0_arr = []
    label0_path = conf.get("DEFAULT", "ELASTIC_LABEL0")
#    label0_path = './benign/elastic_ultrasonic'
    
    for obj in os.listdir(label0_path):
        if os.path.isdir(label0_path+'/'+obj):
            for sub_obj in os.listdir(label0_path+'/'+obj):
                img_arr = io.imread(label0_path+'/'+obj+'/'+sub_obj)
                label0_arr.append(img_arr)
        else:
            img_arr = io.imread(label0_path+'/'+obj)
            label0_arr.append(img_arr)
            
    print ('label0 img total num : '+ str(len(label0_arr)))
    
    label1_arr = []
#    label1_path = './malignant/elastic_ultrasonic'
    label1_path = conf.get("DEFAULT", "ELASTIC_LABEL1")
    for obj in os.listdir(label1_path):
        if os.path.isdir(label1_path+'/'+obj):
            for sub_obj in os.listdir(label1_path+'/'+obj):
                img_arr = io.imread(label1_path+'/'+obj+'/'+sub_obj)
                label1_arr.append(img_arr)
        else:
            img_arr = io.imread(label1_path+'/'+obj)
            label1_arr.append(img_arr)
            
    print ('label1 img total num : '+ str(len(label1_arr)))
    
    label_toal = label0_arr+label1_arr
    
    #####################################################################################
    train_X = np.array(label_toal)
    train_Y = np.hstack((np.zeros(len(label0_arr)),np.ones(len(label1_arr))))
    
    
    num_classes = 2  
    img_rows, img_cols = label_toal[0].shape[0], label_toal[0].shape[1]     # shouble 302x430
    train_X = train_X.reshape(train_X.shape[0], img_rows, img_cols, 3)
    X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.2)    #20% for validtion
    print ('test data num :'+str(X_test.shape[0]))
    
    
    X_train = (X_train/255.0).astype('float32')
    X_test = (X_test/255.0).astype('float32')
    
    input_shape = (img_rows, img_cols, 3)
    y_train = keras.utils.to_categorical(y_train,num_classes)
    y_test  = keras.utils.to_categorical(y_test,num_classes)
    
    
    train_datagen = ImageDataGenerator(
        horizontal_flip = True,
        vertical_flip = True,
        zoom_range = 0.2)
    train_gen = train_datagen.flow(X_train, y_train, batch_size=32, shuffle=True, seed=1)   #gene traindata
    
    valid_datagen = ImageDataGenerator(
        horizontal_flip = True,
        vertical_flip = True,
        zoom_range = 0.2)
    valid_gen = valid_datagen.flow(X_test, y_test, batch_size=32, shuffle=True, seed=1)
    
    model = get_model(input_shape,num_classes)
    
    # 定义损失函数、优化函数和评测方法。
    LearningRate = 0.01
    decay = 0.0001
    n_epochs = 80
    sgd = optimizers.SGD(lr=LearningRate, decay=LearningRate/n_epochs, momentum=0.9, nesterov=True)
    model.compile(loss=keras.losses.binary_crossentropy,optimizer=sgd,metrics=['accuracy'])
    
    file_path = conf.get("DEFAULT", "ELASTIC_CHECKPOINT_PATH")
    file_name = conf.get("DEFAULT", "ELASTIC_MODEL_NAME")
#    file_path = 'elastic_Ultrasonic_Model.h5'
    callbacks_s = get_callbacks(file_path,file_name,model,patience=20)
    
    
    
    model.fit_generator(generator=train_gen, steps_per_epoch=len(X_train)/32, epochs=n_epochs, validation_data=valid_gen, 
                                             validation_steps=32,callbacks = callbacks_s,verbose=1)
    
    score = model.evaluate(X_test,y_test)
    print ('Test loss: ', score[0])
    print ('Test accuracy:', score[1])




