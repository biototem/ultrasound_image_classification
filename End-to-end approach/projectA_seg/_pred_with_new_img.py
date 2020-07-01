import torch
import numpy as np
import os
import glob
import dataset_reader
import imageio
import cv2
import matplotlib.pyplot as plt
import yaml

from a_config import device, seg_net_save_dir, seg_new_img_test_in_dir_prefix, seg_new_img_test_out_dir_prefix, seg_thresh


in_dim = 3
device = torch.device(device)

target_hw = (256, 256)


def main(NetClass, key_name):
    torch.set_grad_enabled(False)
    model_id = NetClass.model_id

    test_in_dir = seg_new_img_test_in_dir_prefix + '_{}'.format(key_name)
    test_out_dir = seg_new_img_test_out_dir_prefix + '_{}'.format(key_name)

    os.makedirs(test_in_dir, exist_ok=True)
    os.makedirs(test_out_dir, exist_ok=True)

    ims_path = []
    for im_path in glob.glob(os.path.join(test_in_dir, '*.*')):
        if os.path.splitext(im_path)[1] in ['.tif']:
            ims_path.append(im_path)

    ck_name = '{}/model_{}_{}.pt'.format(seg_net_save_dir, model_id, key_name)

    net = NetClass(in_dim)
    net.load_state_dict(torch.load(ck_name, map_location='cpu'))
    net = net.to(device)
    net.eval()

    for im_path in ims_path:
        im = imageio.imread(im_path)

        im_name = os.path.splitext(os.path.split(im_path)[1])[0]
        out_im_path = '{}/{}_{}_{}.png'.format(test_out_dir, im_name, model_id, key_name)
        out_data_path = '{}/{}_{}_{}.yml'.format(test_out_dir, im_name, model_id, key_name)

        im = dataset_reader.pad_picture(im, min(target_hw[1], im.shape[1]), min(target_hw[0], im.shape[0]), cv2.INTER_AREA, fill_value=65535)
        im = dataset_reader.center_pad(im, target_hw, 65535)

        batch_im = torch.tensor([im], dtype=torch.float) / 65535
        batch_im = batch_im.permute(0, 3, 1, 2)
        batch_im = batch_im.to(device)

        net_out = net(batch_im)

        # 输出图像
        out = torch.softmax(net_out, 1)[0, 2]
        out_im = (out * 255).cpu().type(torch.uint8).numpy()
        plt.imsave(out_im_path, out_im, cmap=plt.cm.jet)

        # 输出详细信息
        cls_map = net_out.argmax(dim=1)
        bg_pixel_num = (cls_map == dataset_reader.DatasetReader.class2id['bg']).sum().item()
        benign_pixel_num = (cls_map == dataset_reader.DatasetReader.class2id['benign']).sum().item()
        malignant_pixel_num = (cls_map == dataset_reader.DatasetReader.class2id['malignant']).sum().item()

        if malignant_pixel_num + benign_pixel_num == 0:
            pred = 0
        else:
            pred = 1 if malignant_pixel_num / (malignant_pixel_num + benign_pixel_num) > seg_thresh else 0

        out_dict = {
            'pred': pred,
            'bg_pixel_num': bg_pixel_num,
            'benign_pixel_num': benign_pixel_num,
            'malignant_pixel_num': malignant_pixel_num,
        }
        yaml.safe_dump(out_dict, open(out_data_path, 'w'))


if __name__ == '__main__':
    from MainNet1 import MainNet

    main(MainNet, 'u')
    # main(MainNet, 'eu')

    # from MainNet2 import MainNet
    # main(MainNet, 'u')
    # main(MainNet, 'eu')

