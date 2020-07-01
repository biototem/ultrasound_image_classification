import sys
import os
sys.path.append(os.path.join(os.path.split(__file__)[0], '..'))

import torch
import torch.nn.functional as F
import time
import numpy as np
import yaml
from dataset_reader import DatasetReader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from draw_confusion_matrix import draw_confusion_matrix
from tensorboardX import SummaryWriter
import cv2
import threading

from a_config import device, epoch, batch_size, dataset_path, seg_net_save_dir, seg_net_train_logs_dir_prefix, seg_thresh


in_dim = 3
device = torch.device(device)

os.makedirs(seg_net_save_dir, exist_ok=True)


last_time = 0

need_to_show = []
need_to_show_lock = threading.Lock()
need_to_quit = False


def auto_show_thread():
    # 用于训练过程动态观察结果
    while not need_to_quit:
        need_to_show_lock.acquire(True)
        for i, im in enumerate(need_to_show):
            cv2.imshow(str(i), im)
        need_to_show_lock.release()
        cv2.waitKey(5000)


new_thread = threading.Thread(target=auto_show_thread, daemon=True)
new_thread.start()


def test_show(out, batch_cm, batch_im):
    # 用于训练过程动态观察结果
    global last_time
    if time.time() - last_time > 5:
        last_time = time.time()
        with torch.no_grad():
            out = out.cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8) * 100
            batch_cm = (batch_cm.cpu().numpy() * 100).astype(np.uint8)
            batch_im = (batch_im.cpu().permute(0, 2, 3, 1).numpy() * 255.).astype(np.uint8)
            need_to_show_lock.acquire(True)
            need_to_show.clear()
            need_to_show.append(batch_im[0])
            need_to_show.append(batch_cm[0])
            need_to_show.append(out[0][..., 0])

            need_to_show.append(batch_im[-1])
            need_to_show.append(batch_cm[-1])
            need_to_show.append(out[-1][..., 0])
            need_to_show_lock.release()


def main(NetClass, key_name):
    model_id = NetClass.model_id

    train_dataset_path = '{}/{}/train'.format(dataset_path, key_name)
    eval_dataset_path = '{}/{}/eval'.format(dataset_path, key_name)

    ck_name = '{}/model_{}_{}.pt'.format(seg_net_save_dir, model_id, key_name)
    ck_extra_name = '{}/extra_{}_{}.yml'.format(seg_net_save_dir, model_id, key_name)
    cm_name = '{}/cm_valid_{}_{}.png'.format(seg_net_save_dir, model_id, key_name)

    logdir = '{}_{}_{}' .format(seg_net_train_logs_dir_prefix, model_id, key_name)
    sw = SummaryWriter(logdir)

    train_dataset = DatasetReader(train_dataset_path, is_require_cls_blance=True)
    eval_dataset = DatasetReader(eval_dataset_path)

    net = NetClass(in_dim)

    net = net.to(device)

    batch_count = train_dataset.get_batch_count(batch_size)

    optim = torch.optim.Adam(net.parameters(), 1e-3, eps=1e-8)
    optim_adjust = torch.optim.lr_scheduler.MultiStepLR(optim, [90, 180, 270], gamma=0.1)

    max_valid_value = 0.

    class_weight_for_loss = torch.tensor([1, 1, 1], dtype=torch.float, device=device)

    for e in range(epoch):
        net.train()
        optim_adjust.step(e)
        train_acc = 0
        train_loss = 0
        for b in range(batch_count):

            batch_im, batch_cm, batch_cls = train_dataset.get_batch(batch_size)

            batch_im = torch.tensor(batch_im.astype(np.int32), dtype=torch.float) / 65535
            # batch_im += (torch.rand_like(batch_im) * 0.1 - 0.05)
            batch_cm = torch.tensor(batch_cm, dtype=torch.long)

            batch_im = batch_im.permute(0, 3, 1, 2)
            # batch_cm = batch_cm[:, None, :, :]

            batch_im = batch_im.to(device)
            batch_cm = batch_cm.to(device)

            net_out = net(batch_im)

            with torch.no_grad():
                out = torch.argmax(net_out, 1, keepdim=True)
                acc = torch.eq(out, batch_cm).sum(dtype=torch.float) / np.prod(out.shape)
                test_show(out, batch_cm, batch_im)

            loss = F.cross_entropy(net_out, batch_cm, class_weight_for_loss)

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
                    im, cm, cls = eval_dataset.get_im_patch_list_to_combind_predict(i)

                    batch_im = torch.tensor([im], dtype=torch.float) / 65535
                    # batch_cm = torch.tensor([cm])

                    batch_im = batch_im.permute(0, 3, 1, 2)

                    batch_im = batch_im.to(device)
                    # batch_cm = batch_cm.to(device)

                    net_out = net(batch_im)
                    out = torch.argmax(net_out, 1)
                    all_label.append(cls)

                    cls1_pixel_num = (out == 1).sum().item()
                    cls2_pixel_num = (out == 2).sum().item()
                    # all_cls_pixel_num = cls1_pixel_num + cls2_pixel_num

                    if cls2_pixel_num / (cls1_pixel_num + cls2_pixel_num) > seg_thresh:
                        all_pred.append(2)
                    else:
                        all_pred.append(1)

                _accuracy = accuracy_score(all_label, all_pred)
                _malignant_precision, _malignant_recall, _malignant_f1, _ =\
                    precision_recall_fscore_support(all_label, all_pred, pos_label=2, average='binary')

                _benign_precision, _benign_recall, _benign_f1, _ =\
                    precision_recall_fscore_support(all_label, all_pred, pos_label=1, average='binary')

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
                    draw_confusion_matrix(cm, list(eval_dataset.class2id.keys())[1:], cm_name)

            # early exit
            if _accuracy == 1.:
                print('found valid acc == 1. , early exit')
                break

    sw.close()


if __name__ == '__main__':
    from MainNet1 import MainNet
    main(MainNet, 'u')
    # main(MainNet, 'eu')

    # 因为效果不好而被否决
    #from MainNet2 import MainNet
    #main(MainNet, 'u')
    #main(MainNet, 'eu')
