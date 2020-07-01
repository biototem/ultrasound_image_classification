'''
对测试集输出分数
'''

import sys
import os
# sys.path.append(os.path.split(__file__)[0])
sys.path.append(os.path.join(os.path.split(__file__)[0], '..'))

import torch

import numpy as np
import os
from dataset_reader import DatasetReader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from draw_confusion_matrix import draw_confusion_matrix

from a_config import device, dataset_path, simple_net_save_dir_prefix, simple_net_test_out, simple_thresh


in_dim = 3
device = torch.device(device)

test_out_file = simple_net_test_out
test_out = []


def main(NetClass, key_name, scale):
    assert scale in [32, 64, 128]
    torch.set_grad_enabled(False)

    model_id = NetClass.model_id
    
    save_dir = '{}.{}'.format(simple_net_save_dir_prefix, scale)
    os.makedirs(save_dir, exist_ok=True)

    test_dataset_path = '{}/{}/test'.format(dataset_path, key_name)

    ck_name = '{}/model_{}_{}.pt'.format(save_dir, model_id, key_name)
    cm_test_name = '{}/cm_test_{}_{}.png'.format(save_dir, model_id, key_name)

    test_dataset = DatasetReader(test_dataset_path, target_hw=(scale, scale))

    net = NetClass(in_dim)

    net.load_state_dict(torch.load(ck_name, map_location='cpu'))

    net = net.to(device)

    net.eval()

    all_pred = []
    all_label = []

    for i in range(len(test_dataset)):
        ims, cls = test_dataset.get_im_patch_list_to_combind_predict(i, one_im=False)

        batch_im = torch.tensor(ims.astype(np.int32), dtype=torch.float) / 65535
        # batch_cls = torch.tensor([cls]).repeat(len(batch_im))

        batch_im = batch_im.permute(0, 3, 1, 2)

        batch_im = batch_im.to(device)
        # batch_cls = batch_cls.to(device)

        net_out = net(batch_im)
        out = torch.argmax(net_out, 1)
        all_label.append(cls)

        if out.sum(dtype=torch.float).item() > out.shape[0] * simple_thresh:
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
               'b_prec: {:.3f} b_rec: {:.3f} b_f1: {:.3f} model {}_{} x{}'.format(_accuracy,
                                            _malignant_precision, _malignant_recall, _malignant_f1,
                                            _benign_precision, _benign_recall, _benign_f1, model_id, key_name, scale)

    print(out_line)
    test_out.append(out_line)

    cm = confusion_matrix(all_label, all_pred)
    draw_confusion_matrix(cm, list(test_dataset.class2id.keys()), cm_test_name)


if __name__ == '__main__':
    # Net1 因为效果一致而速度较慢而被否决
    # from MainNet1 import MainNet
    # main(MainNet, 'u', 32)
    # main(MainNet, 'eu', 32)
    # main(MainNet, 'u', 64)
    # main(MainNet, 'eu', 64)
    # main(MainNet, 'u', 128)
    # main(MainNet, 'eu', 128)

    # Net2 普超部分因为效果较差而被否决
    from MainNet2 import MainNet
    # main(MainNet, 'u', 32)
    main(MainNet, 'eu', 32)
    # main(MainNet, 'u', 64)
    main(MainNet, 'eu', 64)
    # main(MainNet, 'u', 128)
    main(MainNet, 'eu', 128)

    open(test_out_file, 'w').write('\n'.join(test_out))
