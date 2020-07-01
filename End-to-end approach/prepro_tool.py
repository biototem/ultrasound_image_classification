import numpy as np
import glob
import imageio
import os
import cv2
import myprepro
from typing import Iterable

from sklearn.feature_extraction.image import extract_patches_2d


fill_value = 65535


def center_pad(im, target_hw, fill_value=0):
    pad_h = target_hw[0] - im.shape[0]
    pad_w = target_hw[1] - im.shape[1]
    pad_t = pad_h // 2
    pad_b = pad_h - pad_t
    pad_l = pad_w // 2
    pad_r = pad_w - pad_l
    if pad_h < 0:
        pad_t = pad_b = 0
    if pad_w < 0:
        pad_l = pad_r = 0
    if len(im.shape) > 2 and not isinstance(fill_value, Iterable):
        fill_value = [fill_value] * im.shape[2]
    im = cv2.copyMakeBorder(im, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=fill_value)
    return im


def center_crop(im, target_hw):
    crop_h = im.shape[0] - target_hw[0]
    crop_w = im.shape[1] - target_hw[1]
    crop_t = crop_h // 2
    crop_b = crop_h - crop_t
    crop_l = crop_w // 2
    crop_r = crop_h - crop_l
    if crop_h < 0:
        crop_t = crop_b = None
    if crop_w < 0:
        crop_l = crop_r = None
    im = im[crop_t:im.shape[0]-crop_b, crop_l:im.shape[1]-crop_r]
    return im


def random_crop(im, target_hw, return_crop_params=False):
    crop_h = im.shape[0]-target_hw[0]
    crop_w = im.shape[1]-target_hw[1]
    start_y = np.random.randint(0, crop_h) if crop_h > 0 else 0
    start_x = np.random.randint(0, crop_w) if crop_w > 0 else 0
    im = im[start_y:start_y+target_hw[0], start_x:start_x+target_hw[1]]
    crop_params = {
        'x1': start_x,
        'x2': start_x+target_hw[1],
        'y1': start_y,
        'y2': start_y+target_hw[0]
    }
    if return_crop_params:
        return im, crop_params
    else:
        return im


def pad_picture(img, width, height, interpolation=cv2.INTER_NEAREST, fill_value=0.):
    """
    padded picture to specified shape, then return this and padded mask
    :param img: input numpy array
    :param width: output image width
    :param height: output image height
    :param interpolation: control img interpolation
    :param fill_value: control background fill value
    :return: output numpy array
    """
    s_height, s_width = img.shape[:2]
    new_shape = [height, width]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    # ratio = s_width / s_height
    width_prop = width / s_width
    height_prop = height / s_height
    min_prop = min(width_prop, height_prop)
    img = cv2.resize(img, (int(s_width * min_prop), int(s_height * min_prop)), interpolation=interpolation)
    img_start_x = width / 2 - s_width * min_prop / 2
    img_start_y = height / 2 - s_height * min_prop / 2
    if isinstance(fill_value, Iterable):
        new_img = np.zeros(new_shape, dtype=img.dtype)
        new_img[:, :] = np.asarray(fill_value)
    else:
        new_img = np.full(new_shape, fill_value, img.dtype)
    new_img[int(img_start_y):int(img_start_y)+img.shape[0], int(img_start_x):int(img_start_x)+img.shape[1]] = img
    return new_img


def crop(im, x1, x2, y1, y2):
    im = im[y1:y2, x1:x2]
    return im


def random_rotate(im, angle=None, fill_value=0):
    center_yx = (im.shape[0]//2, im.shape[1]//2)
    R = np.eye(3)
    if angle is None:
        angle = np.random.uniform(-180, 180)
    R[:2] = cv2.getRotationMatrix2D(angle=-angle, center=center_yx[::-1], scale=1.)
    im = cv2.warpPerspective(im, R, dsize=im.shape[:2][::-1], flags=cv2.INTER_AREA, borderValue=fill_value)
    return im


def my_extract_patches_2d(im, patch_hw):
    assert im.shape[0] % patch_hw[0] == 0 and im.shape[1] % patch_hw[1] == 0
    half_step_h = patch_hw[0] // 2
    half_step_w = patch_hw[1] // 2
    im_patches = []
    for half_h in range(im.shape[0] // half_step_h - 1):
        for half_w in range(im.shape[1] // half_step_w - 1):
            im_patches.append(im[half_h*half_step_h: half_h*half_step_h+patch_hw[0],
                                 half_w*half_step_w: half_w*half_step_w+patch_hw[1]])
    return np.asarray(im_patches, np.uint16)

