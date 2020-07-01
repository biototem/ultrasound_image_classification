'''
要求数据集格式
数据集根--医生名--良恶性类别--图像文件

注意，图像文件名必须为 XXX-T.???
其中 XXX必须为数字，T必须为1或2，???必须为tif

数据集构造第一步
将数据集内数据按普超和彩超分离到俩个文件夹中
'''


import os
import shutil

from a_config import ori_dataset_path as data_dir
from a_config import out_dir1 as out_dir


for doctor in os.listdir(data_dir):
    doctor_dir = os.path.join(data_dir, doctor)
    out_doctor_dir = os.path.join(out_dir, doctor)

    if not os.path.isdir(doctor_dir):
        continue
    
    for cls in os.listdir(doctor_dir):
        cls_dir = os.path.join(doctor_dir, cls)
        out_cls_dir = os.path.join(out_doctor_dir, cls)
        
        out_cls_ultrasonic_dir = os.path.join(out_cls_dir, 'ultrasonic')
        out_cls_elastic_ultrasonic_dir = os.path.join(out_cls_dir, 'elastic_ultrasonic')
        
        os.makedirs(out_cls_ultrasonic_dir, exist_ok=True)
        os.makedirs(out_cls_elastic_ultrasonic_dir, exist_ok=True)
        
        for im_file in os.listdir(cls_dir):
            full_im_file = os.path.join(cls_dir, im_file)
            if os.path.splitext(im_file)[0][-1] == '1':
                target_file = os.path.join(out_cls_elastic_ultrasonic_dir)
            elif os.path.splitext(im_file)[0][-1] == '2':
                target_file = os.path.join(out_cls_ultrasonic_dir)
            else:
                raise AssertionError("Wrong file name", im_file)

            target_file = os.path.join(target_file, im_file)

            shutil.copy(full_im_file, target_file)
