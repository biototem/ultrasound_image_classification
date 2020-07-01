# -*- coding: utf-8 -*-
# @Licence : bio-totem
"""
Created on Fri Aug 24 09:06:11 2018

@author: zengyouling
"""

import os
import cv2
from skimage import io
import numpy as np

def rgbcrop_resize(benign_read_path,malignant_read_path):
    #BreastGray    BreastRGB
    benign_ultrasonic_write_path =  os.path.join(benign_read_path,"..","resize","benign_ultrasonic")
    if not os.path.exists(benign_ultrasonic_write_path): os.makedirs(benign_ultrasonic_write_path)
    benign_elastic_ultrasonic_write_path =  os.path.join(benign_read_path,"..","resize","benign_elastic")
    if not os.path.exists(benign_elastic_ultrasonic_write_path): os.makedirs(benign_elastic_ultrasonic_write_path)
    malignant_ultrasonic_write_path =  os.path.join(malignant_read_path,"..","resize","malignant_ultrasonic")
    if not os.path.exists(malignant_ultrasonic_write_path): os.makedirs(malignant_ultrasonic_write_path)
    malignant_elastic_ultrasonic_write_path = os.path.join(malignant_read_path,"..","resize","malignant_elastic_ultrasonic")
    if not os.path.exists(malignant_elastic_ultrasonic_write_path): os.makedirs(malignant_elastic_ultrasonic_write_path)
    
    img_row = [];img_col = []
    
    for imgName in os.listdir(benign_read_path):
        #img = cv2.imread(benign_read_path+'\\'+imgName)
        #img_arr = np.array(img) 
        img_arr = io.imread(os.path.join(benign_read_path,imgName))
        if imgName == '129-2.tif':
            img_arr = img_arr[0:117,68:456]
            index = np.argwhere(img_arr == [65535,65535,65535])
            for i in range(len(index)):
                img_arr[index[i][0],index[i][1]] = [0,0,0]
                
            img_arr = cv2.resize(img_arr,(430,302),interpolation= cv2.INTER_CUBIC)
            
            #cv2.imwrite(benign_ultrasonic_write_path+'\\'+imgName.split('.')[0]+'.png',img_arr)
            io.imsave(os.path.join(benign_ultrasonic_write_path,imgName.split('.')[0]+'.png'),img_arr)
            continue
        
        #locate box
        point = 0 ; offet = 0
        row_range = []
        for row in range(img_arr.shape[0]):
            if np.sum(img_arr[row,:]) < img_arr.shape[1]*65535*3:
                point = row
                offet += 1
        row_range.append(point-offet); row_range.append(point)
        point = 0 ; offet = 0
        col_range = []
        for col in range(img_arr.shape[1]):
            if np.sum(img_arr[:,col]) < img_arr.shape[0]*65535*3:
                point = col
                offet += 1
        col_range.append(point-offet); col_range.append(point)  
        
        #crop
        img_arr = img_arr[row_range[0]:row_range[1], col_range[0]:col_range[1]]
        
        #replace white background
        index = np.argwhere(img_arr == [65535,65535,65535])
        for i in range(len(index)):
            img_arr[index[i][0],index[i][1]] = [0,0,0]
        
        img_arr = cv2.resize(img_arr,(430,302),interpolation= cv2.INTER_CUBIC)
        
        if imgName.split('.')[0].endswith('1'):  
            io.imsave(os.path.join(benign_elastic_ultrasonic_write_path,imgName.split('.')[0]+'.png'),img_arr)
            #cv2.imwrite(benign_elastic_ultrasonic_write_path+'\\'+imgName,img_arr)
        else:
            io.imsave(os.path.join(benign_ultrasonic_write_path,imgName.split('.')[0]+'.png'),img_arr)
            #cv2.imwrite(benign_ultrasonic_write_path+'\\'+imgName,img_arr)
        
        img_row.append(img_arr.shape[0])
        img_col.append(img_arr.shape[1])
        
    #malignant_imglist = []
    for imgName in os.listdir(malignant_read_path):
        #img = cv2.imread(malignant_read_path+'\\'+imgName)
        #img_arr = np.array(img)
        img_arr = io.imread(os.path.join(malignant_read_path,imgName))
        
        #locate box
        point = 0 ; offet = 0
        row_range = []
        for row in range(img_arr.shape[0]):
            if np.sum(img_arr[row,:]) < img_arr.shape[1]*65535*3:
                point = row
                offet += 1
        row_range.append(point-offet); row_range.append(point)    
        point = 0 ; offet = 0
        col_range = []
        for col in range(img_arr.shape[1]):
            if np.sum(img_arr[:,col]) < img_arr.shape[0]*65535*3:
                point = col
                offet += 1
        col_range.append(point-offet); col_range.append(point)
        
        #crop
        img_arr = img_arr[row_range[0]:row_range[1], col_range[0]:col_range[1]]
        
        #replace white background
        index = np.argwhere(img_arr == [65535,65535,65535])
        for i in range(len(index)):
            img_arr[index[i][0],index[i][1]] = [0,0,0]
        try :
            img_arr = cv2.resize(img_arr,(430,302),interpolation= cv2.INTER_CUBIC)
            
            if imgName.split('.')[0].endswith('1'):
                io.imsave(os.path.join(malignant_elastic_ultrasonic_write_path,imgName.split('.')[0]+'.png'),img_arr)
                #cv2.imwrite(malignant_elastic_ultrasonic_write_path+'\\'+imgName.split('.')[0]+'.png',img_arr)
            else:
                io.imsave(os.path.join(malignant_ultrasonic_write_path,imgName.split('.')[0]+'.png'),img_arr)
            #cv2.imwrite(malignant_ultrasonic_write_path+'\\'+imgName.split('.')[0]+'.png',img_arr)
        except Exception:
            print(imgName)
        
        img_row.append(img_arr.shape[0])
        img_col.append(img_arr.shape[1])
    
    rowMax = max(img_row);colMax = max(img_col)
    print (rowMax,colMax)

if __name__ == '__main__':
    import configparser
    conf = configparser.ConfigParser()
    current_path = os.path.dirname(__file__)
    conf.read(os.path.join(current_path,"sys.ini")) 
    dir_sub = conf.get("DEFAULT", "IMG_SOURCE_DIR")
    for doctor_name in os.listdir(dir_sub):
        benign_read_path = os.path.join(dir_sub,doctor_name,'benign')
        malignant_read_path = os.path.join(dir_sub,doctor_name,'malignant')
        rgbcrop_resize(benign_read_path,malignant_read_path)
