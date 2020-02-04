import sys
import os
sys.path.append(os.path.join(os.path.split(__file__)[0], '..'))

import torch

import os
from dataset_reader import DatasetReader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from draw_confusion_matrix import draw_confusion_matrix

from a_config import device, seg_net_save_dir, dataset_path, seg_thresh


in_dim = 3
device = torch.device(device)

test_out_file = '{}/test.txt'.format(seg_net_save_dir)
test_out = []

os.makedirs(seg_net_save_dir, exist_ok=True)

# dataset_type in {'train', 'eval', 'test'}


def main(NetClass, key_name, dataset_type='test'):
    torch.set_grad_enabled(False)

    model_id = NetClass.model_id

    test_dataset_path = '{}/{}/{}'.format(dataset_path, key_name, dataset_type)
    test_dataset = DatasetReader(test_dataset_path)

    ck_name = '{}/model_{}_{}.pt'.format(seg_net_save_dir, model_id, key_name)
    cm_test_name = '{}/cm_{}_{}_{}.png'.format(seg_net_save_dir, dataset_type, model_id, key_name)

    net = NetClass(in_dim)
    net.load_state_dict(torch.load(ck_name))
    net = net.to(device)
    net.eval()

    all_pred = []
    all_label = []

    for i in range(len(test_dataset)):
        im, cm, cls = test_dataset.get_im_patch_list_to_combind_predict(i)

        batch_im = torch.tensor([im], dtype=torch.float) / 65535
        batch_im = batch_im.permute(0, 3, 1, 2)
        batch_im = batch_im.to(device)

        net_out = net(batch_im)
        out = torch.argmax(net_out, 1)
        all_label.append(cls)

        cls1_pixel_num = (out == 1).sum().item()
        cls2_pixel_num = (out == 2).sum().item()

        if cls1_pixel_num + cls2_pixel_num == 0:
            all_pred.append(1)
        else:
            if cls2_pixel_num / (cls1_pixel_num + cls2_pixel_num) > seg_thresh:
                all_pred.append(2)
            else:
                all_pred.append(1)

    _accuracy = accuracy_score(all_label, all_pred)
    _malignant_precision, _malignant_recall, _malignant_f1, _ = \
        precision_recall_fscore_support(all_label, all_pred, pos_label=2, average='binary')

    _benign_precision, _benign_recall, _benign_f1, _ = \
        precision_recall_fscore_support(all_label, all_pred, pos_label=1, average='binary')

    _accuracy = float(_accuracy)
    _malignant_precision = float(_malignant_precision)
    _malignant_recall = float(_malignant_recall)
    _malignant_f1 = float(_malignant_f1)
    _benign_precision = float(_benign_precision)
    _benign_recall = float(_benign_recall)
    _benign_f1 = float(_benign_f1)

    out_line = '{} acc: {:.3f} m_prec: {:.3f} m_rec: {:.3f} m_f1: {:.3f} '\
               'b_prec: {:.3f} b_rec: {:.3f} b_f1: {:.3f} model {}_{}'.format(dataset_type, _accuracy,
                                                    _malignant_precision, _malignant_recall, _malignant_f1,
                                                    _benign_precision, _benign_recall, _benign_f1, model_id, key_name)

    print(out_line)
    test_out.append(out_line)

    cm = confusion_matrix(all_label, all_pred)
    draw_confusion_matrix(cm, list(test_dataset.class2id.keys())[1:], cm_test_name)


if __name__ == '__main__':
    from MainNet1 import MainNet
    main(MainNet, 'u')
    # main(MainNet, 'eu')

    # from MainNet2 import MainNet
    # main(MainNet, 'u')
    # main(MainNet, 'eu')

    open(test_out_file, 'w').write('\n'.join(test_out))
