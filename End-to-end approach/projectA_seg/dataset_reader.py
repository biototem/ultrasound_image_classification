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
    class2id = {'bg': 0, 'benign': 1, 'malignant': 2}
    id2class = dict(zip(class2id.values(), class2id.keys()))
    
    def __init__(self, dataset_path='./out_dir3/eu/train', target_hw=(256, 256), cm_hw=(128, 128), use_rescale=False,
                 use_patch=False, is_require_cls_blance=False, is_random_rescale_or_patch=False):
        self.target_hw = target_hw
        self.cm_hw = cm_hw
        self.use_rescale = use_rescale
        self.use_patch = use_patch
        self.is_require_cls_blance = is_require_cls_blance
        self.is_random_rescale_or_patch = is_random_rescale_or_patch

        self.im_list = []
        self.im_benign_list = []
        self.im_malignant_list = []
        
        for i, f in enumerate(glob.glob('{}/**/*.tif'.format(dataset_path), recursive=True)):
            mask_path = os.path.splitext(f)[0] + '.png'
            mask = imageio.imread(mask_path)
            cls_name = os.path.split(os.path.split(f)[0])[1]
            cls_id = self.class2id[cls_name]
            new_mask = np.where(mask > 0, np.full_like(mask, cls_id), np.full_like(mask, self.class2id['bg']))
            item = [f, new_mask, cls_id]
            self.im_list.append(item)
            if cls_id == self.class2id['benign']:
                self.im_benign_list.append(i)
            elif cls_id == self.class2id['malignant']:
                self.im_malignant_list.append(i)
            else:
                raise AssertionError('Unknow label')
    
    def im_enchance(self):
        pass
    
    def get_item(self, i, use_prepro=True, need_im_name=False):
        
        break_count = 0
        
        while True:
            break_count += 1
            impath, cm, cls = self.im_list[i]
            # calc pure black pixel percent greater than 90%, need to skip this.
            im = imageio.imread(impath)
            #cm = imageio.imread(cmpath)
            im_name = os.path.split(impath)[1]

            if use_prepro:
                crop_size_h = np.random.randint(48, self.target_hw[0])
                crop_size_w = np.random.randint(48 , self.target_hw[1])
                im, crop_params = random_crop(im, (crop_size_h, crop_size_w), return_crop_params=True)
                cm = crop(cm, **crop_params)
                if np.random.uniform() > 0.5:
                    im = center_pad(im, self.target_hw, fill_value)
                    cm = center_pad(cm, self.target_hw, self.class2id['bg'])
                else:
                    inter = np.random.choice([cv2.INTER_AREA, cv2.INTER_LINEAR])
                    im = cv2.resize(im, self.target_hw, interpolation=inter)
                    cm = cv2.resize(cm, self.target_hw, interpolation=cv2.INTER_NEAREST)
                im, affine_params = myprepro.random_affine(im, None, degrees=(-170, 170), translate=(.1, .1), scale=(.7, 1.5), borderValue=(fill_value, fill_value, fill_value))
                affine_params['flags'] = cv2.INTER_NEAREST
                affine_params['borderValue'] = (0, 0, 0)
                cm = cv2.warpPerspective(cm, **affine_params)
                # im = random_rotate(im, fill_value=fill_value)

                cm = cv2.resize(cm, self.cm_hw[::-1], interpolation=cv2.INTER_NEAREST)

            # check
            check_zero = np.where(im == np.full_like(im, 65535), np.ones_like(im), np.zeros_like(im))
            if check_zero.sum() / np.prod(check_zero.shape) < 0.9 or break_count > 100:
                break
            else:
                #print('too many black pixel')
                pass
        
        if need_im_name:
            return im_name, im, cm, cls
        
        return im, cm, cls
    
    def get_batch_count(self, batch_size):
        return int(np.ceil(len(self.im_list) / batch_size))

    def get_batch(self, batch_size):
        ims = []
        cms = []
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
            im, cm, cls = self.get_item(i)
            ims.append(im)
            cms.append(cm)
            clss.append(cls)

        ims = np.asarray(ims)
        cms = np.asarray(cms)
        clss = np.asarray(clss)
        
        return ims, cms, clss

    def get_im_patch_list_to_combind_predict(self, i, resize=False, need_im_name=False):
        if need_im_name:
            im_name, im, cm, cls = self.get_item(i, use_prepro=False, need_im_name=need_im_name)
        else:
            im, cm, cls = self.get_item(i, use_prepro=False, need_im_name=need_im_name)
        if resize:
            im = pad_picture(im, self.target_hw[1], self.target_hw[0], cv2.INTER_AREA, fill_value=65535)
            cm = pad_picture(cm, self.cm_hw[1], self.cm_hw[0], cv2.INTER_NEAREST, fill_value=0)
        else:
            im = pad_picture(im, min(self.target_hw[1], im.shape[1]), min(self.target_hw[0], im.shape[0]), cv2.INTER_AREA, fill_value=65535)
            cm = pad_picture(cm, min(self.target_hw[1], cm.shape[1]), min(self.target_hw[0], cm.shape[0]), cv2.INTER_AREA, fill_value=0)
            im = center_pad(im, self.target_hw, 65535)
            cm = center_pad(cm, self.target_hw, 0)
            cm = cv2.resize(cm, self.cm_hw, interpolation=cv2.INTER_AREA)
            
        if need_im_name:
            return im_name, im, cm, cls
        
        return im, cm, cls
        """
        if one_im:
            im = cv2.resize(im, self.target_hw, interpolation=cv2.INTER_AREA)
            im_patch_list = np.array([im])
        else:
            new_h = int(np.ceil(im.shape[0] / self.target_hw[0]) * self.target_hw[0])
            new_w = int(np.ceil(im.shape[1] / self.target_hw[1]) * self.target_hw[1])
            im = center_pad(im, (new_h, new_w))

            im_patch_list = my_extract_patches_2d(im, self.target_hw)
        
            # im_patch_list = extract_patches_2d(im, self.target_hw)
        return im_patch_list, cls
        """
    
    def __len__(self):
        return len(self.im_list)


if __name__ == '__main__':
    ds = DatasetReader('./out_dir3/u/train', is_require_cls_blance=True)
    
    # test
    #for i in range(len(ds)):
    #    im, cm, cls = ds.get_item(i)
    #    cv2.imshow(str(cls)+'im', im[..., ::-1])
    #    cv2.imshow(str(cls)+'map', cm*120)
    #    cv2.waitKey(0)
    
    # test get_batch
    '''
    bs = 10
    for b in range(ds.get_batch_count(bs)):
        ims, cms, clss = ds.get_batch(bs)
        for im, cm, cls in zip(ims, cms, clss):
            cv2.imshow(str(cls)+'im', im[..., ::-1])
            cv2.imshow(str(cls)+'map', cm * 100)
            cv2.waitKey(0)
    '''
    # test get_im_patch_list_to_combind_predict
    #for it in range(len(ds)):
    #    im, cm, cls = ds.get_im_patch_list_to_combind_predict(it)
    #    cv2.imshow(str(cls)+'im', im)
    #    cv2.imshow(str(cls)+'map', cm*100)
    #    cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    # test get_im_patch_list_to_combind_predict
    for it in range(len(ds)):
        im_name, im, cm, cls = ds.get_im_patch_list_to_combind_predict(it, need_im_name=True)
        cv2.imshow(str(cls)+'im'+im_name, im)
        cv2.imshow(str(cls)+'map'+im_name, cm*100)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
