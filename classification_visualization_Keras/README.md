# 基于Keras的ELASTIC/ULTRASONIC分类模型的特征提取及梯度可视化程序
本程序针对BENIGN和MALIGNANT的B超图像,实现从采样→训练→特征提取→梯度可视化的真个过程的功能。


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


**GPU_USE**是运行项目代码时，所指定占用的GPU，如指定使用第0张，可以设置为GPU_USE = 0。

**ULTRASONIC_CHECKPOINT_PATH**是ULTRASONIC模型训练的checkpoint保存路径,TrainModel和ExtraFea脚本会用到

**ULTRASONIC_MODEL_NAME**是ULTRASONIC模型保存的文件名

**ELASTIC_CHECKPOINT_PATH**是ELASTIC模型训练的checkpoint保存路径,TrainModel和ExtraFea脚本会用到

**ELASTIC_MODEL_NAME**是ELASTIC模型保存的文件名

**IMG_SOURCE_DIR** 原图像主目录,grad_cam_plot和RGBcrop脚本会用到

**BENIGN_SAMPLE_ROOT_DIR** BENIGN图像采样后的图像保存路径,RGBcrop脚本会用到

**MALIGNANT_SAMPLE_ROOT_DIR** MALIGNANT图像采样后的图像保存路径,RGBcrop脚本会用到

## Run
上述sys.ini文件配置好之后,分别执行脚本就可以运行程序,不需要传入任何参数。**util_script.py**脚本封装了keras上建立模型和get_callback方法,不需要单独执行
### RGBcrop_resize_method
```
python RGBcrop_resize_method.py
```

该脚本能将B超原图以中心区域进行采样再resize成302×430大小,用于进行下一步的训练

### *TrainModel_RGB_Resize302x430.py
```
python elastic_Ultrasonic_TrainModel_RGB_Resize302x430.py

python Ultrasonic_TrainModel_RGB_Resize302x430.py

```

上述两个脚本是分别进行elastic和Ultrasonic的训练

### ExtraFea_method.py
```
python ExtraFea_method.py

```

该脚本用于调用elastic/Ultrasonic模型生成图像特征

### grad_cam_plot.py
```
python grad_cam_plot.py

```
该脚本用于调用elastic/Ultrasonic模型的max_pooling2d_4/max_pooling2d_2层进行梯度可视化
