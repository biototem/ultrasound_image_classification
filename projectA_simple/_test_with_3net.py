'''
联合3个不同的尺度的网络对测试集进行预测
'''

import sys
import os
sys.path.append(os.path.join(os.path.split(__file__)[0], '..'))

import torch
import numpy as np
from dataset_reader import DatasetReader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from draw_confusion_matrix import draw_confusion_matrix

from a_config import device, dataset_path, simple_net_3_merge_test_out, simple_net_save_dir_prefix,\
    simple_net_3_merge_test_cm_prefix, simple_thresh, simple_merge_thresh


in_dim = 3
device = torch.device(device)
epoch = 200
batch_size = 64

test_net3_out_file = simple_net_3_merge_test_out
test_out = []


def main(NetClass, key_name):
    torch.set_grad_enabled(False)

    model_id = NetClass.model_id

    test_dataset_path = '{}/{}/test'.format(dataset_path, key_name)

    ck_32_name = '{}.32/model_{}_{}.pt'.format(simple_net_save_dir_prefix, model_id, key_name)
    ck_64_name = '{}.64/model_{}_{}.pt'.format(simple_net_save_dir_prefix, model_id, key_name)
    ck_128_name = '{}.128/model_{}_{}.pt'.format(simple_net_save_dir_prefix, model_id, key_name)

    cm_net3_test_name = '{}_{}_{}.png'.format(simple_net_3_merge_test_cm_prefix, model_id, key_name)
    os.makedirs(os.path.split(cm_net3_test_name)[0], exist_ok=True)

    test_dataset_32 = DatasetReader(test_dataset_path, target_hw=(32, 32))
    test_dataset_64 = DatasetReader(test_dataset_path, target_hw=(64, 64))
    test_dataset_128 = DatasetReader(test_dataset_path, target_hw=(128, 128))

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

    all_pred = []
    all_label = []

    for i in range(len(test_dataset_32)):
        ims_32, cls_32 = test_dataset_32.get_im_patch_list_to_combind_predict(i, one_im=False)
        ims_64, cls_64 = test_dataset_64.get_im_patch_list_to_combind_predict(i, one_im=False)
        ims_128, cls_128 = test_dataset_128.get_im_patch_list_to_combind_predict(i, one_im=False)

        assert cls_32 == cls_64 == cls_128

        tmp_x = [[net_32, ims_32, cls_32], [net_64, ims_64, cls_64], [net_128, ims_128, cls_128]]
        tmp_y = []

        for net, ims, cls in tmp_x:
            batch_im = torch.tensor(ims.astype(np.int32), dtype=torch.float) / 65535
            # batch_cls = torch.tensor([cls]).repeat(len(batch_im))

            batch_im = batch_im.permute(0, 3, 1, 2)

            batch_im = batch_im.to(device)
            # batch_cls = batch_cls.to(device)

            net_out = net(batch_im)
            out = torch.argmax(net_out, 1)

            if out.sum(dtype=torch.float).item() > out.shape[0] * simple_thresh:
                tmp_y.append(1)
            else:
                tmp_y.append(0)

        all_label.append(tmp_x[0][-1])

        if np.sum(tmp_y) > simple_merge_thresh:
            all_pred.append(1)
        else:
            all_pred.append(0)

    _accuracy = accuracy_score(all_label, all_pred)
    _malignant_precision, _malignant_recall, _malignant_f1, _ = \
        precision_recall_fscore_support(all_label, all_pred, pos_label=1, average='binary')

    _benign_precision, _benign_recall, _benign_f1, _ = \
        precision_recall_fscore_support(all_label, all_pred, pos_label=0, average='binary')

    _accuracy = float(_accuracy)
    _malignant_precision = float(_malignant_precision)
    _malignant_recall = float(_malignant_recall)
    _malignant_f1 = float(_malignant_f1)
    _benign_precision = float(_benign_precision)
    _benign_recall = float(_benign_recall)
    _benign_f1 = float(_benign_f1)

    out_line = 'test acc: {:.3f} m_prec: {:.3f} m_rec: {:.3f} m_f1: {:.3f} '\
               'b_prec: {:.3f} b_rec: {:.3f} b_f1: {:.3f} model {}_{}'.format(_accuracy,
                                                    _malignant_precision, _malignant_recall, _malignant_f1,
                                                    _benign_precision, _benign_recall, _benign_f1, model_id, key_name)

    print(out_line)
    test_out.append(out_line)

    cm = confusion_matrix(all_label, all_pred)
    draw_confusion_matrix(cm, list(test_dataset_32.class2id.keys()), cm_net3_test_name)


if __name__ == '__main__':
    # from MainNet1 import MainNet
    # main(MainNet, 'u')
    # main(MainNet, 'eu')

    from MainNet2 import MainNet
    # main(MainNet, 'u')
    main(MainNet, 'eu')

    os.makedirs(os.path.split(test_net3_out_file)[0], exist_ok=True)
    open(test_net3_out_file, 'w').write('\n'.join(test_out))
