import os
import numpy as np
import imageio
import cv2
import glob

from a_config import out_dir2
from a_config import out_dir3
from a_config import fill_value


os.makedirs(out_dir3, exist_ok=True)

for im_path in glob.glob('{}/**/*.tif'.format(out_dir2), recursive=True):

    im = imageio.imread(im_path)

    im_path = im_path.replace(out_dir2, out_dir3, 1)
    os.makedirs(os.path.split(im_path)[0], exist_ok=True)
    mask_path = os.path.splitext(im_path)[0] + '.png'

    mask = np.where(im == np.full_like(im, 65535),
                    np.zeros_like(im), np.ones_like(im))
    mask = np.all(mask, 2).astype(np.uint8)
    r = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = r[0]
    max_contour = None
    max_area = 0
    for c in contours:
        if cv2.contourArea(c) > max_area:
            max_contour = c
            max_area = cv2.contourArea(c)
    boundingbox = cv2.boundingRect(max_contour)
    mask = cv2.drawContours(np.zeros_like(mask), [max_contour], -1, 1, thickness=cv2.FILLED)
    mask = np.where(mask, np.full_like(mask, 255), np.zeros_like(mask))
    im2 = np.where(np.tile(mask[..., None], [1,1,3]) > 0, im, np.full_like(im, fill_value))
    crop_im = im2[boundingbox[1]:boundingbox[1]+boundingbox[3], boundingbox[0]:boundingbox[0]+boundingbox[2]]
    crop_mask = mask[boundingbox[1]:boundingbox[1]+boundingbox[3], boundingbox[0]:boundingbox[0]+boundingbox[2]]
    imageio.imwrite(im_path, crop_im)
    imageio.imwrite(mask_path, crop_mask)
    cv2.imshow("abc", mask)
    cv2.imshow("abc2", im2[..., ::-1])
    cv2.imshow("abc3", crop_im[..., ::-1])
    cv2.imshow("abc4", crop_mask)
    cv2.waitKey(16)

cv2.destroyAllWindows()
