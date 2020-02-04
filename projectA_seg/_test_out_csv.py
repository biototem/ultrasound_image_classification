'''
输出所有数据集所有数据的结果，并存入一个csv文件
csv 格式
数据集类型，图像标签，恶性概率，标签（0代表良性，1代表恶性）
'''

import torch
import numpy as np
import os
from dataset_reader import DatasetReader

from a_config import device, dataset_path, seg_net_save_dir, seg_net_all_result_csv_prefix


in_dim = 3
device = torch.device(device)
epoch = 200
batch_size = 64

save_dir = 'save'
os.makedirs(save_dir, exist_ok=True)


# dict map
# dataset_type -> f_id -> pred32x, pred64x, pred128x, class
big_dict = {}


def main(NetClass, key_name):
    torch.set_grad_enabled(False)

    target_hw=(256, 256)
    model_id = NetClass.model_id

    train_dataset_path = '{}/{}/train'.format(dataset_path, key_name)
    eval_dataset_path = '{}/{}/eval'.format(dataset_path, key_name)
    test_dataset_path = '{}/{}/test'.format(dataset_path, key_name)
    
    ck_name = '{}/model_{}_{}.pt'.format(seg_net_save_dir, model_id, key_name)

    train_dataset = DatasetReader(train_dataset_path, target_hw=target_hw)
    eval_dataset = DatasetReader(eval_dataset_path, target_hw=target_hw)
    test_dataset = DatasetReader(test_dataset_path, target_hw=target_hw)
        
    net = NetClass(in_dim)
    net.load_state_dict(torch.load(ck_name, map_location='cpu'))
    net = net.to(device)
    net.eval()

    for dataset_type_id, dataset in enumerate([train_dataset, eval_dataset, test_dataset]):
        dataset_type = ['train', 'eval', 'test'][dataset_type_id]

        if dataset_type not in big_dict:
            big_dict[dataset_type] = {}
        
        for i in range(len(dataset)):
            im_name, im, cm, cls = dataset.get_im_patch_list_to_combind_predict(i, need_im_name=True)
            
            im_id = os.path.splitext(im_name)[0]
            if im_id not in big_dict[dataset_type]:
                big_dict[dataset_type][im_id] = {}
            
            batch_im = torch.tensor([im], dtype=torch.float) / 65535
            # batch_cm = torch.tensor([cm])
    
            batch_im = batch_im.permute(0, 3, 1, 2)
    
            batch_im = batch_im.to(device)
            # batch_cm = batch_cm.to(device)
    
            net_out = net(batch_im)
            out = torch.argmax(net_out, 1)
    
            cls1_pixel_num = (out == 1).sum(dtype=torch.float).item()
            cls2_pixel_num = (out == 2).sum(dtype=torch.float).item()

            if cls1_pixel_num + cls2_pixel_num == 0:
                out_pred = 0
            else:
                out_pred = cls2_pixel_num / (cls1_pixel_num + cls2_pixel_num)
            
            assert not np.isnan(out_pred)
            
            if 'pred' not in big_dict[dataset_type][im_id]:
                big_dict[dataset_type][im_id]['pred'] = out_pred
            else:
                raise AssertionError('Error, found pred setting in 2 times')
                
            if 'class' not in big_dict[dataset_type][im_id]:
                big_dict[dataset_type][im_id]['class'] = cls - 1
            else:
                assert big_dict[dataset_type][im_id]['class'] == cls


def write_dict_to_csv_and_clear_dict(test_out_file):
    lines = []
    for dataset_type in big_dict:
        for im_id in big_dict[dataset_type]:
            line = [dataset_type, im_id,
                    big_dict[dataset_type][im_id]['pred'],
                    big_dict[dataset_type][im_id]['class']]
            line = ','.join([str(i) for i in line])
            lines.append(line)
    
    csv = '\n'.join(lines)
    os.makedirs(os.path.split(test_out_file)[0], exist_ok=True)
    open(test_out_file, 'w').write(csv)
    big_dict.clear()


if __name__ == '__main__':
    from MainNet1 import MainNet
    main(MainNet, 'u')
    
    test_out_file = '{}_{}_{}.csv'.format(seg_net_all_result_csv_prefix, MainNet.model_id, 'u')
    write_dict_to_csv_and_clear_dict(test_out_file)
    
    # main(MainNet, 'eu')
    #
    # test_out_file = 'prob_net_{}_{}.csv'.format(MainNet.model_id, 'eu')
    # write_dict_to_csv_and_clear_dict(test_out_file)
    #
    # from MainNet2 import MainNet
    # main(MainNet, 'u')
    #
    # test_out_file = 'prob_net_{}_{}.csv'.format(MainNet.model_id, 'u')
    # write_dict_to_csv_and_clear_dict(test_out_file)
    #
    # main(MainNet, 'eu')
    #
    # test_out_file = 'prob_net_{}_{}.csv'.format(MainNet.model_id, 'eu')
    # write_dict_to_csv_and_clear_dict(test_out_file)
