'''
使用训练集对网络进行训练，每训练3epoch后使用验证集进行验证
'''

import sys
import os
sys.path.append(os.path.join(os.path.split(__file__)[0], '..'))

import torch
import torch.nn.functional as F

import numpy as np
import os
import yaml
from tensorboardX import SummaryWriter
from dataset_reader import DatasetReader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from draw_confusion_matrix import draw_confusion_matrix

from a_config import device, dataset_path, simple_net_save_dir_prefix, simple_net_train_logs_dir_prefix, simple_thresh
from a_config import epoch, batch_size


in_dim = 3
device = torch.device(device)


def main(NetClass, key_name, scale=32):
    torch.set_grad_enabled(True)
    
    assert scale in [32, 64, 128]
    model_id = NetClass.model_id
    
    save_dir = '{}.{}'.format(simple_net_save_dir_prefix, scale)
    os.makedirs(save_dir, exist_ok=True)

    train_dataset_path = '{}/{}/train'.format(dataset_path, key_name)
    eval_dataset_path = '{}/{}/eval'.format(dataset_path, key_name)

    ck_name = '{}/model_{}_{}.pt'.format(save_dir, model_id, key_name)
    ck_extra_name = '{}/extra_{}_{}.yml'.format(save_dir, model_id, key_name)
    cm_name = '{}/cm_valid_{}_{}.png'.format(save_dir, model_id, key_name)

    logdir = '{}_{}_{}.{}' .format(simple_net_train_logs_dir_prefix, model_id, key_name, scale)
    sw = SummaryWriter(logdir)

    train_dataset = DatasetReader(train_dataset_path, is_require_cls_blance=True, target_hw=(scale, scale))
    eval_dataset = DatasetReader(eval_dataset_path, target_hw=(scale, scale))

    net = NetClass(in_dim)

    net = net.to(device)

    batch_count = train_dataset.get_batch_count(batch_size)

    optim = torch.optim.Adam(net.parameters(), 1e-3, eps=1e-8)
    optim_adjust = torch.optim.lr_scheduler.MultiStepLR(optim, [90, 180, 270], gamma=0.1)

    max_valid_value = 0.

    class_weight_for_loss = torch.tensor([1, 1], dtype=torch.float, device=device)

    for e in range(epoch):
        net.train()
        optim_adjust.step(e)
        train_acc = 0
        train_loss = 0
        for b in range(batch_count):

            batch_im, batch_cls = train_dataset.get_batch(batch_size)

            batch_im = torch.tensor(batch_im.astype(np.int32), dtype=torch.float) / 65535
            # batch_im += (torch.rand_like(batch_im) * 0.1 - 0.05)
            batch_cls = torch.tensor(batch_cls, dtype=torch.long)

            batch_im = batch_im.permute(0, 3, 1, 2)

            batch_im = batch_im.to(device)
            batch_cls = batch_cls.to(device)

            net_out = net(batch_im)
            # net_out = net_train(batch_im)

            with torch.no_grad():
                out = torch.argmax(net_out, 1)
                acc = torch.eq(out, batch_cls).sum(dtype=torch.float) / len(out)

            loss = F.cross_entropy(net_out, batch_cls, class_weight_for_loss)

            train_acc += acc.item()
            train_loss += loss.item()

            print('epoch: {} train acc: {:.3f} loss: {:.3f}'.format(e, acc.item(), loss.item()))
            optim.zero_grad()
            loss.backward()
            optim.step()

        train_acc = train_acc / batch_count
        train_loss = train_loss / batch_count

        sw.add_scalar('train_acc', train_acc, global_step=e)
        sw.add_scalar('train_loss', train_loss, global_step=e)

        # here to check eval
        if (e+1) % 3 == 0:
            with torch.no_grad():
                net.eval()

                all_pred = []
                all_label = []

                for i in range(len(eval_dataset)):
                    ims, cls = eval_dataset.get_im_patch_list_to_combind_predict(i, one_im=False)

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
                _malignant_precision, _malignant_recall, _malignant_f1, _ =\
                    precision_recall_fscore_support(all_label, all_pred, pos_label=1, average='binary')

                _benign_precision, _benign_recall, _benign_f1, _ =\
                    precision_recall_fscore_support(all_label, all_pred, pos_label=0, average='binary')

                _accuracy = float(_accuracy)
                _malignant_precision = float(_malignant_precision)
                _malignant_recall = float(_malignant_recall)
                _malignant_f1 = float(_malignant_f1)
                _benign_precision = float(_benign_precision)
                _benign_recall = float(_benign_recall)
                _benign_f1 = float(_benign_f1)

                sw.add_scalar('eval_acc', _accuracy, global_step=e)
                sw.add_scalar('eval_m_prec', _malignant_precision, global_step=e)
                sw.add_scalar('eval_m_recall', _malignant_recall, global_step=e)
                sw.add_scalar('eval_m_f1', _malignant_f1, global_step=e)
                sw.add_scalar('eval_b_prec', _benign_precision, global_step=e)
                sw.add_scalar('eval_b_recall', _benign_recall, global_step=e)
                sw.add_scalar('eval_b_f1', _benign_f1, global_step=e)

                print('epoch: {} eval acc: {:.3f} m_prec: {:.3f} m_rec: {:.3f} m_f1: {:.3f} '
                      'b_prec: {:.3f} b_rec: {:.3f} b_f1: {:.3f}'.format(e, _accuracy,
                                                            _malignant_precision, _malignant_recall, _malignant_f1,
                                                            _benign_precision, _benign_recall, _benign_f1))

                avg_f1 = (_malignant_f1 + _benign_f1) / 2

                #if _benign_precision - _malignant_precision > 0.2:
                #    class_weight_for_loss[1] += 0.1

                if avg_f1 >= max_valid_value:
                    max_valid_value = avg_f1
                    torch.save(net.state_dict(), ck_name)
                    extra = {
                        'accuracy': _accuracy,
                        'm_precision': _malignant_precision,
                        'm_recall': _malignant_recall,
                        'm_f1': _malignant_f1,
                        'b_precision': _benign_precision,
                        'b_recall': _benign_recall,
                        'b_f1': _benign_f1,
                    }
                    yaml.safe_dump(extra, open(ck_extra_name, 'w'))
                    cm = confusion_matrix(all_label, all_pred)
                    draw_confusion_matrix(cm, list(eval_dataset.class2id.keys()), cm_name)

            # early exit
            if _accuracy == 1.:
                print('found valid acc == 1. , early exit')
                break

    sw.close()


if __name__ == '__main__':
    # from MainNet1 import MainNet
    # main(MainNet, 'u', 32)
    # main(MainNet, 'eu', 32)
    # main(MainNet, 'u', 64)
    # main(MainNet, 'eu', 64)
    # main(MainNet, 'u', 128)
    # main(MainNet, 'eu', 128)

    from MainNet2 import MainNet
    # main(MainNet, 'u', 32)
    main(MainNet, 'eu', 32)
    # main(MainNet, 'u', 64)
    main(MainNet, 'eu', 64)
    # main(MainNet, 'u', 128)
    main(MainNet, 'eu', 128)
