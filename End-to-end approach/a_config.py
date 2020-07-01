'''
简写解释
u 普超
eu 彩超
cm 混淆矩阵
lr 学习率
prefix 命名前缀

'''

import os

# base setting
## 项目根目录路径，一般不需要修改
project_root = os.path.split(__file__)[0]
print(project_root)

## 指定运行在哪个设备
device = 'cuda:0'
## 训练参数，分别是学习率，训练轮数，每批量大小
train_lr = 1e-3
epoch = 400
batch_size = 48


# dataset setting
## 用于图像背景填充色，如果要修改，则需要重新构造数据集
fill_value = 65535
## 原始数据集和原始标签的路径，注意格式
## 数据集格式 数据集根目录/医生名/良恶性/图像
## 标签格式 是否属于测试集，文件ID，是否为恶性
ori_dataset_path = project_root + '/breastSWE'
ori_label_path = ori_dataset_path + '/ultrasound_grouping.csv'
## 构造数据集时存放中间数据的目录，其中 out_dir3 为最终数据集构造完成时所存放的目录
out_dir1 = project_root + '/out_dir1'
out_dir2 = project_root + '/out_dir2'
out_dir3 = project_root + '/out_dir3'
## 指向已完成的数据集
dataset_path = out_dir3


# project seg setting

# 恶性像素/(恶性像素+良性像素) > seg_thresh 时，结果判定为良性
seg_thresh = 0.5

## 指定模型保存位置和日志保存位置
seg_net_save_dir = project_root + '/projectA_seg/save'
seg_net_train_logs_dir_prefix = project_root + '/projectA_seg/logs'
## 测试输出
### 该文件储存着数据集内每个样本的结果
seg_net_all_result_csv_prefix = project_root + '/projectA_seg/prob_net'
## 用于预测新的图像
### 将新图像放入此文件夹
seg_new_img_test_in_dir_prefix = project_root + '/projectA_seg/test_in'
### 对新图像分析后的结果将存入此文件夹
seg_new_img_test_out_dir_prefix = project_root + '/projectA_seg/test_out'


# project simple setting

## 恶性判断百分比，用于输出每个尺度网络的预测结果
simple_thresh = 0.5
## 恶性判断数量 > simple_merge_thresh 时，结果判定为恶性
## 范围 0 <= simple_merge_thresh < 3
simple_merge_thresh = 1

## 指定模型保存位置和日志保存位置
simple_net_save_dir_prefix = project_root + '/projectA_simple/save'
simple_net_train_logs_dir_prefix = project_root + '/projectA_simple/logs'
## 测试输出
simple_net_test_out = project_root + '/projectA_simple/test.txt'
simple_net_all_result_csv_prefix = project_root + '/projectA_simple/prob_net'
simple_net_3_merge_test_out = project_root + '/projectA_simple/test_net_3_merge.txt'
simple_net_3_merge_test_cm_prefix = project_root + '/projectA_simple/cm_test_net_3_merge'
## 用于预测新的图像
### 将新图像放入此文件夹
simple_new_img_test_in_dir_prefix = project_root + '/projectA_simple/test_in'
### 对新图像分析后的结果将存入此文件夹
simple_new_img_test_out_dir_prefix = project_root + '/projectA_simple/test_out'
