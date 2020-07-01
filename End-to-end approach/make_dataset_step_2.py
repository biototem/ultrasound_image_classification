import os
import shutil
import csv
import numpy as np

from a_config import out_dir1
from a_config import out_dir2
from a_config import ori_label_path as label_file


eval_percent = 0.3

label_ori = list(csv.reader(open(label_file, 'r')))
label_dict = {}

# 字典格式
# file_id -> is_elastic -> is_test

for la in label_ori:
    f_id = int(la[1])
    f_is_test = not bool(int(la[0]))
    f_is_malignant = bool(int(la[2]))

    if f_id not in label_dict:
        label_dict[f_id] = {}
    if f_is_malignant not in label_dict[f_id]:
        label_dict[f_id][f_is_malignant] = None

    label_dict[f_id][f_is_malignant] = f_is_test


eu_train_dataset_path = os.path.join(out_dir2, 'eu', 'train')
eu_eval_dataset_path = os.path.join(out_dir2, 'eu', 'eval')
eu_test_dataset_path = os.path.join(out_dir2, 'eu', 'test')

u_train_dataset_path = os.path.join(out_dir2, 'u', 'train')
u_eval_dataset_path = os.path.join(out_dir2, 'u', 'eval')
u_test_dataset_path = os.path.join(out_dir2, 'u', 'test')

os.makedirs(os.path.join(eu_train_dataset_path, 'benign'), exist_ok=True)
os.makedirs(os.path.join(eu_eval_dataset_path, 'benign'), exist_ok=True)
os.makedirs(os.path.join(eu_test_dataset_path, 'benign'), exist_ok=True)
os.makedirs(os.path.join(eu_train_dataset_path, 'malignant'), exist_ok=True)
os.makedirs(os.path.join(eu_eval_dataset_path, 'malignant'), exist_ok=True)
os.makedirs(os.path.join(eu_test_dataset_path, 'malignant'), exist_ok=True)
os.makedirs(os.path.join(u_train_dataset_path, 'benign'), exist_ok=True)
os.makedirs(os.path.join(u_eval_dataset_path, 'benign'), exist_ok=True)
os.makedirs(os.path.join(u_test_dataset_path, 'benign'), exist_ok=True)
os.makedirs(os.path.join(u_train_dataset_path, 'malignant'), exist_ok=True)
os.makedirs(os.path.join(u_eval_dataset_path, 'malignant'), exist_ok=True)
os.makedirs(os.path.join(u_test_dataset_path, 'malignant'), exist_ok=True)

test_elastic_ultrasonic_file_list = []
test_ultrasonic_file_list = []

train_elastic_ultrasonic_file_list = []
train_ultrasonic_file_list = []

eval_elastic_ultrasonic_file_list = []
eval_ultrasonic_file_list = []

for doctor in os.listdir(out_dir1):
    doctor_dir = os.path.join(out_dir1, doctor)

    for cls in os.listdir(doctor_dir):
        cls_dir = os.path.join(doctor_dir, cls)

        for ultra_cls in os.listdir(cls_dir):
            ultra_cls_dir = os.path.join(cls_dir, ultra_cls)

            for name in os.listdir(ultra_cls_dir):
                file = os.path.join(ultra_cls_dir, name)

                f_first_name = os.path.splitext(name)[0]
                f_id = int(f_first_name.split('-')[0])

                f_is_malignant = cls == 'malignant'
                f_is_elastic = ultra_cls == 'elastic_ultrasonic'

                try:
                    label_dict[f_id][f_is_malignant]
                except KeyError:
                    print(f_id, f_is_malignant)
                    continue

                if label_dict[f_id][f_is_malignant]:
                    #
                    item = [file, ultra_cls, cls]
                    if f_is_elastic:
                        test_elastic_ultrasonic_file_list.append(item)
                    else:
                        test_ultrasonic_file_list.append(item)
                else:
                    item = [file, ultra_cls, cls]
                    if f_is_elastic:
                        train_elastic_ultrasonic_file_list.append(item)
                    else:
                        train_ultrasonic_file_list.append(item)

# first process test dataset
for i, it in enumerate(test_elastic_ultrasonic_file_list):
    i = '%d_' % i
    f_name = os.path.split(it[0])[1]
    f_new_path = os.path.join(eu_test_dataset_path, it[2], i + f_name)
    shutil.copy(it[0], f_new_path)

for i, it in enumerate(test_ultrasonic_file_list):
    i = '%d_' % i
    f_name = os.path.split(it[0])[1]
    f_new_path = os.path.join(u_test_dataset_path, it[2], i + f_name)
    shutil.copy(it[0], f_new_path)

# second split some train item to eval
np.random.shuffle(train_elastic_ultrasonic_file_list)
np.random.shuffle(train_ultrasonic_file_list)

eu_eval_ids = np.arange(int(len(train_elastic_ultrasonic_file_list) * eval_percent))
u_eval_ids = np.arange(int(len(train_ultrasonic_file_list) * eval_percent))

for id_ in sorted(eu_eval_ids)[::-1]:
    eval_elastic_ultrasonic_file_list.append(train_elastic_ultrasonic_file_list[id_])
    del train_elastic_ultrasonic_file_list[id_]

for id_ in sorted(u_eval_ids)[::-1]:
    eval_ultrasonic_file_list.append(train_ultrasonic_file_list[id_])
    del train_ultrasonic_file_list[id_]

# third copy file to special dir. Train
for i, it in enumerate(train_elastic_ultrasonic_file_list):
    i = '%d_' % i
    f_name = os.path.split(it[0])[1]
    f_new_path = os.path.join(eu_train_dataset_path, it[2], i + f_name)
    shutil.copy(it[0], f_new_path)

for i, it in enumerate(train_ultrasonic_file_list):
    i = '%d_' % i
    f_name = os.path.split(it[0])[1]
    f_new_path = os.path.join(u_train_dataset_path, it[2], i + f_name)
    shutil.copy(it[0], f_new_path)

# fourth copy file to special dir. Eval
for i, it in enumerate(eval_elastic_ultrasonic_file_list):
    i = '%d_' % i
    f_name = os.path.split(it[0])[1]
    f_new_path = os.path.join(eu_eval_dataset_path, it[2], i + f_name)
    shutil.copy(it[0], f_new_path)

for i, it in enumerate(eval_ultrasonic_file_list):
    i = '%d_' % i
    f_name = os.path.split(it[0])[1]
    f_new_path = os.path.join(u_eval_dataset_path, it[2], i + f_name)
    shutil.copy(it[0], f_new_path)
