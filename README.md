# Breast Cancer Ultrasound Image Classification 

projectA_seg: Based on segmentation methods by counting the proportion of pixels classified as malignant
projectA_simple: Based on classification methods by directly classifying images as benign or malignant

### Abbreviation used
u Ultrasound
eu Elastograpy Ultrasound  
cm Confusion Matrix
lr Learning rate  

# Dependencies
```
pytorch >= 1.1  
numpy  
tensorboardX  
scikit-image  
opencv-python >= 4.0  
seaborn  
matplotlib  
imageio  
```

# Quick Guide

## 1. Training  
    projectA_seg/_train.py 
    The trained model with highest F1 score on validation set will be saved in /save folder together with relevant information and confusion matrix
    
    projectA_simple/_train.py  
    three different scales of image are used, thus three different best train models are saved in /save.32 /save.64 /save.128  

## 2. Testing  
    run /_test.py to test images in the test set    

## 3. Predciting a new image
    put the image in /test_in 
    run _pred_with_new_img.py  
    output saved in /test_out  
    
    output format:    
    
    projectA_seg   
    benign_pixel_num: number of pixels classified as benign  
    bg_pixel_num: number of pixels classified as background
    malignant_pixel_num:  number of pixels classified as malignant  
    pred: 0 - benign，1 - malignant  
    
    projectA_simple
    pred:  0 - benign，1 - malignant   
    pred_x32: scale 32 network prediction result 
    pred_x64: scale 64 network prediction result   
    pred_x128 scale 128 network prediction result   
    prob_x32: scale 32 network output probability   
    prob_x64: scale 64 network output probability   
    prob_x128: scale 128 network output probability   

