# -*- coding: utf-8 -*-
# @Licence : bio-totem
"""
Created on Wed Aug 29 09:18:04 2018

@author: zengyouling
"""
import os
import numpy as np
from skimage import io
from keras.models import load_model
from keras.models import Model
import configparser

if __name__ == "__main__":
    conf = configparser.ConfigParser()
    current_path = os.path.dirname(__file__)
    conf.read(os.path.join(current_path,"sys.ini"))    
    
    GPU_USE = conf.get("DEFAULT", "GPU_USE")
    #DEVICE = '\"/gpu:' + GPU_USE + '\"'
    print('\"/gpu:' + GPU_USE + '\"')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(GPU_USE))
    print('\"CUDA_VISIBLE_DEVICES\"'+ ' = ' '\"'+ GPU_USE + '\"')
    
    ultrasonic_type_lst = ["ELASTIC","ULTRASONIC"]
    for ultrasonic_type in ultrasonic_type_lst:
        
        model_path = conf.get("DEFAULT", ultrasonic_type + "_CHECKPOINT_PATH")
        file_name = conf.get("DEFAULT", ultrasonic_type+ "_MODEL_NAME")
        model = load_model(os.path.join(model_path,file_name))
        mymodel = load_model(model_path)
        
        # =============================================================================
        #model_path = conf.get("DEFAULT", "ULTRASONIC_CHECKPOINT_PATH")
        #file_name = conf.get("DEFAULT", "ULTRASONIC_MODEL_NAME")
        #model = load_model(os.path.join(model_path,file_name))
        #mymodel = load_model(model_path)
        # =============================================================================
        
        dense2_model_layer = Model(inputs = mymodel.input, outputs = mymodel.get_layer('Dense-2').output)
        dense1_model_layer = Model(inputs = mymodel.input, outputs = mymodel.get_layer('Dense-1').output)
        
        file_namelist = []
        label0_arr = []
        save_root_dir = conf.get("DEFAULT", "BENIGN_SAMPLE_ROOT_DIR")
        label0_path = os.path.join(save_root_dir,ultrasonic_type.lower())
        for obj in os.listdir(label0_path):
            if os.path.isdir(label0_path+'/'+obj):
                continue
            else:
                img_arr = io.imread(label0_path+'/'+obj)
                label0_arr.append(img_arr)
                file_namelist.append(label0_path.split('/')[-1] +'/'+obj)
        print ('label0 img need extract feature number is : ' + str(len(label0_arr)))   
        
        label1_arr = []
        save_root_dir = conf.get("DEFAULT", "MALIGNANT_SAMPLE_ROOT_DIR")
        label1_path = os.path.join(save_root_dir,ultrasonic_type.lower())
        for obj in os.listdir(label1_path):
            if os.path.isdir(label1_path+'/'+obj):
                continue
            else:
                img_arr = io.imread(label1_path+'/'+obj)
                label1_arr.append(img_arr)
                file_namelist.append(label1_path.split('/')[-1] +'/'+obj)
        print ('label1 img need extract feature number is : ' + str(len(label1_arr)))
        
        all_img_arr = np.array(label0_arr+label1_arr)
        row_size,col_size,chan = label0_arr[0].shape
        all_img_arr = all_img_arr.reshape(all_img_arr.shape[0],row_size,col_size,chan)
        all_img_arr = (all_img_arr/255.0).astype('float32')
        
        
        dense2_layer_fea = dense2_model_layer.predict(all_img_arr)
        print (ultrasonic_type.lower() + ' dense2 output fea dims : '+ str(dense2_layer_fea.shape))
        
        dense1_layer_fea = dense1_model_layer.predict(all_img_arr)
        print (ultrasonic_type.lower() + ' dense1 output fea dims : '+ str(dense1_layer_fea.shape))
        
        fp = open(os.path.join(save_root_dir,'Feature_'+ultrasonic_type.lower()+'.csv','w'))
        for i in range(dense2_layer_fea.shape[0]):
            fp.write(file_namelist[i]+',')
            fp.write(','.join (str(x) for x in dense2_layer_fea[i].tolist())+',')
            fp.write(','.join (str(x) for x in dense1_layer_fea[i].tolist())+'\n')
        fp.close()
        
        print (ultrasonic_type.lower() + ' dense2 fea number recoding: ' + str(dense2_layer_fea.shape[0]))
        print (ultrasonic_type.lower() + ' dense1 fea number recoding: ' + str(dense1_layer_fea.shape[0]))