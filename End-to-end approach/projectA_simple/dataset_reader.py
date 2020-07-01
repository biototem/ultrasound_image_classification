import sys
import os
sys.path.append(os.path.join(os.path.split(__file__)[0], '..'))
import numpy as np
import glob
import imageio
import cv2
import myprepro
from typing import Iterable
from prepro_tool import *


class DatasetReader:
    class2id = {'benign': 0, 'malignant': 1}
    id2class = dict(zip(class2id.values(), class2id.keys()))
    
    def __init__(self, dataset_path='./out_dir2/eu/train', target_hw=(32, 32), use_rescale=False,
                 use_patch=False, is_require_cls_blance=False, is_random_rescale_or_patch=False):
        self.target_hw = target_hw
        self.use_rescale = use_rescale
        self.use_patch = use_patch
        self.is_require_cls_blance = is_require_cls_blance
        self.is_random_rescale_or_patch = is_random_rescale_or_patch

        self.im_list = []
        self.im_benign_list = []
        self.im_malignant_list = []
        
        for i, f in enumerate(glob.glob('{}/**/*.tif'.format(dataset_path), recursive=True)):
            cls = os.path.split(os.path.split(f)[0])[1]
            item = [f, self.class2id[cls]]
            self.im_list.append(item)
            if item[1] == 0:
                self.im_benign_list.append(i)
            else:
                self.im_malignant_list.append(i)
    
    def im_enchance(self):
        pass
    
    def get_item(self, i, use_prepro=True, need_im_name=False):
        break_try = 0
        while True:
            break_try+=1
            impath, cls = self.im_list[i]
            # calc pure black pixel percent greater than 90%, need to skip this.
            im = imageio.imread(impath)
            
            im_name = os.path.split(impath)[1]

            if use_prepro:
                #crop_size_h = np.random.randint(128, self.target_hw[0])
                #crop_size_w = np.random.randint(128, self.target_hw[1])
                #if np.random.uniform() > 0.3:
                #    im = center_pad(im, self.target_hw, fill_value)
                #else:
                #    inter = np.random.choice([cv2.INTER_AREA, cv2.INTER_LINEAR])
                #    im = cv2.resize(im, self.target_hw, interpolation=inter)
                #im = myprepro.random_affine(im, None, degrees=(-170, 170), translate=(.1, .1), scale=(.9, 1.2), borderValue=(fill_value, fill_value, fill_value))
                # im = random_rotate(im, fill_value=fill_value)
                if im.shape[0] < self.target_hw[0] or im.shape[1] < self.target_hw[1]:
                    im = center_pad(im, target_hw=self.target_hw, fill_value=(fill_value, fill_value, fill_value))
                im = random_crop(im, (self.target_hw[0], self.target_hw[1]))
                rot_code = np.random.choice([-1, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180])
                if rot_code != -1:
                    im = cv2.rotate(im, rot_code)

                # check
                check_zero = np.where(im == np.full_like(im, fill_value), np.ones_like(im), np.zeros_like(im))
                if check_zero.sum() / np.prod(check_zero.shape) < 0.5 or break_try < 20:
                    break
                else:
                    #print('too many black pixel')
                    pass
            else:
                break
            assert im.shape[0] == self.target_hw[0] and im.shape[1] == self.target_hw[1]
        if need_im_name:
            return im_name, im, cls
        return im, cls
    
    def get_batch_count(self, batch_size):
        return int(np.ceil(len(self.im_list) / batch_size))

    def get_batch(self, batch_size):
        ims = []
        clss = []
        ids = []
        if self.is_require_cls_blance:
            assert batch_size > 1
            bs1 = batch_size // 2
            np.random.shuffle(self.im_benign_list)
            ids.extend(self.im_benign_list[:bs1])
            
            bs2 = batch_size - bs1
            np.random.shuffle(self.im_malignant_list)
            ids.extend(self.im_malignant_list[:bs2])
        else:
            ids = np.arange(len(self.im_list))
            np.random.shuffle(ids)
            ids = ids[:batch_size]
        for i in ids:
            im, cls = self.get_item(i)
            ims.append(im)
            clss.append(cls)
            
        ims = np.asarray(ims, np.uint16)
        clss = np.asarray(clss, np.int)
        
        return ims, clss

    def get_im_patch_list_to_combind_predict(self, i, one_im=False, need_im_name=False):
        if need_im_name:
            im_name, im, cls = self.get_item(i, use_prepro=False, need_im_name=need_im_name)
        else:
            im, cls = self.get_item(i, use_prepro=False, need_im_name=need_im_name)
            
        if one_im:
            im = cv2.resize(im, self.target_hw, interpolation=cv2.INTER_AREA)
            im_patch_list = np.array([im])
        else:
            new_h = int(np.ceil(im.shape[0] / self.target_hw[0]) * self.target_hw[0])
            new_w = int(np.ceil(im.shape[1] / self.target_hw[1]) * self.target_hw[1])
            im = center_pad(im, (new_h, new_w), (fill_value, fill_value, fill_value))

            im_patch_list = my_extract_patches_2d(im, self.target_hw)
        
            # im_patch_list = extract_patches_2d(im, self.target_hw)
        if need_im_name:
            return im_name, im_patch_list, cls
        return im_patch_list, cls
    
    def __len__(self):
        return len(self.im_list)


if __name__ == '__main__':
    ds = DatasetReader('./out_dir2/u/train', (128,128), is_require_cls_blance=True)
    
    # test
#    for it in range(len(ds)):
#        im, cls = ds.get_item(it)
#        cv2.imshow(str(cls), im[..., ::-1])
#        cv2.waitKey(0)
    
    # test get_batch
    bs = 10
    for b in range(ds.get_batch_count(bs)):
        ims, clss = ds.get_batch(bs)
        for im, cls in zip(ims, clss):
            cv2.imshow(str(cls), im[..., ::-1])
            cv2.waitKey(0)
    
    '''
    # test get_im_patch_list_to_combind_predict
    for it in range(len(ds)):
        im_patch_list, cls = ds.get_im_patch_list_to_combind_predict(it)
        for im in im_patch_list:
            cv2.imshow(str(cls), im[..., ::-1])
            cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''