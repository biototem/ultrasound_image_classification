# feature extraction from B mode and SWE ultrasound images of breast cancer patients
The programs includes image sampling->training->feature extraction->gradinet visualization


##  requirements

- tensorflow		1.7.0
- keras		2.2.4
- numpy		1.14.5
- opencv-python 
- scikit-learn 0.19.1

## Configure

```cp sys.ini.example sys.ini```

```gedit sys.ini```  # set the configuration

#### [DEFAULT]


**GPU_USE** default GPU_USE = 0。

**ULTRASONIC_CHECKPOINT_PATH** B mode ultrasound model checkpoint saved path, used in TrainModel and ExtraFea scripts

**ULTRASONIC_MODEL_NAME**  B mode ultraosound model checkpoint name

**ELASTIC_CHECKPOINT_PATH** SWE ultrasound model checkpoint saved path, used in TrainModel and ExtraFea scripts

**ELASTIC_MODEL_NAME** SWE ultraosound model checkpoint name

**IMG_SOURCE_DIR** image path, used in grad_cam_plot and RGBcrop scripts

**BENIGN_SAMPLE_ROOT_DIR** saved path of patches sampled from BENIGN images, used in RGBcrop script

**MALIGNANT_SAMPLE_ROOT_DIR** aved path of patches sampled from MALIGNANT images, used in RGBcrop script 

## Run
After the above sys.ini file is configured, you can run the program by executing the scripts separately without passing in any parameters. 
**util_script.py**The callback method does not need to be executed separately.

### RGBcrop_resize_method
```
python RGBcrop_resize_method.py
```
The script samples from the center of the original ultrasound image and then resize it to 302×430 for the next training.


### *TrainModel_RGB_Resize302x430.py
```
python elastic_Ultrasonic_TrainModel_RGB_Resize302x430.py

python Ultrasonic_TrainModel_RGB_Resize302x430.py

```
model training

### ExtraFea_method.py
```
python ExtraFea_method.py

```
feature extraction

### grad_cam_plot.py
```
python grad_cam_plot.py

```
visualization heatmap from max_pooling2d_4/max_pooling2d_2 layers
