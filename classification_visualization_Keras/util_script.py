# -*- coding: utf-8 -*-
# @Licence : bio-totem
"""
Created on Mon Aug 30 11:39:07 2018

@author: Bohrium.Kwong
"""

import os
import time
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D,Dropout

class Mycbk(ModelCheckpoint):
    def __init__(self, model, filepath ,monitor = 'val_loss',mode='min', save_best_only=True):
        self.single_model = model
        super(Mycbk,self).__init__(filepath, monitor, save_best_only, mode)
    def set_model(self,model):
        super(Mycbk,self).set_model(self.single_model)

def get_callbacks(filepath,file_name,model,patience=10):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    if not os.path.exists(filepath): os.makedirs(filepath)
    msave = Mycbk(model, os.path.join(filepath,file_name))
    
    file_dir = os.path.join(filepath,'logs')#+ time.strftime('%Y_%m_%d',time.localtime(time.time()))
    if not os.path.exists(file_dir): os.makedirs(file_dir)
    tb_log = TensorBoard(log_dir=file_dir)
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, 
                                  patience=2, verbose=0, mode='min', epsilon=-0.95, cooldown=0, min_lr=1e-8)
    log_cv = CSVLogger(os.path.join(filepath,time.strftime('%Y_%m_%d',time.localtime(time.time())) +'_log.csv'), separator=',', append=True)
    return [es, msave,reduce_lr,tb_log,log_cv]


def get_model(input_shape,num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten(name='Dense-4'))
    # =============================================================================
    # model.add(Dense(1024,activation='relu',name='Dense-3'))
    # model.add(Dropout(0.5))
    # =============================================================================
    model.add(Dense(512, activation='relu',name='Dense-2'))
    model.add(Dropout(0.5))
    model.add(Dense(256,activation='relu',name='Dense-1'))   
    model.add(Dense(num_classes, activation='sigmoid',name='Dense-0'))
    return model