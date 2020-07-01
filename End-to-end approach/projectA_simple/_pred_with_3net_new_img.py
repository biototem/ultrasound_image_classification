'''
对新图像进行预测
test_in_u test_in_eu 文件夹输入新图像
test_out_u test_out_eu 文件夹得到结果
结果保存为 yml 格式，可以用文本编辑器打开查看
结果文件名格式如下
XXX_I_M.txt
XXX代表图像名
I，代表模型编号
M，如果是eu，代表是彩超，如果是u，代表普超
包含如下结果
pred：3个网络合并的预测标签
pred_x32: 尺度为32的网络的预测标签
pred_x64: 尺度为64的网络的预测标签
pred_x128: 尺度为128的网络的预测标签
prob_x32: 尺度为32的网络的预测概率
prob_x64: 尺度为64的网络的预测概率
prob_x128: 尺度为128的网络的预测概率
'''

import sys
import os
sys.path.append(os.path.join(os.path.split(__file__)[0], '..'))

import torch
import numpy as np
import os
import glob
import imageio
import yaml

from prepro_tool import center_pad, my_extract_patches_2d
from a_config import device, fill_value, simple_net_save_dir_prefix,\
    simple_new_img_test_in_dir_prefix, simple_new_img_test_out_dir_prefix


in_dim = 3
device = torch.device(device)


def extract_patch(im: np.ndarray, patch_hw):
    new_h = int(np.ceil(im.shape[0] / patch_hw[0]) * patch_hw[0])
    new_w = int(np.ceil(im.shape[1] / patch_hw[1]) * patch_hw[1])
    im = center_pad(im, (new_h, new_w), (fill_value, fill_value, fill_value))

    im_patch_list = my_extract_patches_2d(im, patch_hw)
    return im_patch_list


def main(NetClass, key_name):
    torch.set_grad_enabled(False)

    model_id = NetClass.model_id

    test_data_in_dir = '{}_{}'.format(simple_new_img_test_in_dir_prefix, key_name)
    os.makedirs(test_data_in_dir, exist_ok=True)
    test_data_out_dir = '{}_{}'.format(simple_new_img_test_out_dir_prefix, key_name)
    os.makedirs(test_data_out_dir, exist_ok=True)

    ck_32_name = '{}.32/model_{}_{}.pt'.format(simple_net_save_dir_prefix, model_id, key_name)
    ck_64_name = '{}.64/model_{}_{}.pt'.format(simple_net_save_dir_prefix, model_id, key_name)
    ck_128_name = '{}.128/model_{}_{}.pt'.format(simple_net_save_dir_prefix, model_id, key_name)

    net_32 = NetClass(in_dim)
    net_64 = NetClass(in_dim)
    net_128 = NetClass(in_dim)

    net_32.load_state_dict(torch.load(ck_32_name, map_location=torch.device('cpu')))
    net_64.load_state_dict(torch.load(ck_64_name, map_location=torch.device('cpu')))
    net_128.load_state_dict(torch.load(ck_128_name, map_location=torch.device('cpu')))

    net_32 = net_32.to(device)
    net_64 = net_64.to(device)
    net_128 = net_128.to(device)

    net_32.eval()
    net_64.eval()
    net_128.eval()

    for i, im_path in enumerate(glob.glob(os.path.join(test_data_in_dir, '*.tif'))):
        name = os.path.splitext(os.path.split(im_path)[1])[0]
        out_data_path = os.path.join(test_data_out_dir, '{}_{}_{}.yml'.format(name, model_id, key_name))

        print(im_path, out_data_path, sep=' -> ')

        im = imageio.imread(im_path)
        ims_32 = extract_patch(im, (32, 32))
        ims_64 = extract_patch(im, (64, 64))
        ims_128 = extract_patch(im, (128, 128))

        tmp_x = [[net_32, ims_32], [net_64, ims_64], [net_128, ims_128]]
        tmp_y = []
        tmp_prob = []

        for net, ims in tmp_x:
            batch_im = torch.tensor(ims.astype(np.int32), dtype=torch.float) / 65535

            batch_im = batch_im.permute(0, 3, 1, 2)
            batch_im = batch_im.to(device)

            net_out = net(batch_im)
            out = torch.argmax(net_out, 1)

            malignant_prob_out = torch.softmax(net_out, 1)[:, 1]
            tmp_prob.append(malignant_prob_out.mean().item())

            if out.sum(dtype=torch.float).item() > out.shape[0] * 0.5:
                tmp_y.append(1)
            else:
                tmp_y.append(0)

        if np.sum(tmp_y) > 1:
            pred = 1
        else:
            pred = 0

        out_dict = {
            'pred': pred,
            'pred_x32': tmp_y[0],
            'pred_x64': tmp_y[1],
            'pred_x128': tmp_y[2],
            'prob_x32': tmp_prob[0],
            'prob_x64': tmp_prob[1],
            'prob_x128': tmp_prob[2],
        }

        yaml.safe_dump(out_dict, open(out_data_path, 'w'))


if __name__ == '__main__':
    # from MainNet1 import MainNet
    # main(MainNet, 'u')
    # main(MainNet, 'eu')

    from MainNet2 import MainNet
    # main(MainNet, 'u')
    main(MainNet, 'eu')
