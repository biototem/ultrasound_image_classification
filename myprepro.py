import cv2
import numpy as np
import random
import math


# 空变换矩阵
# def new_mat():
#     return np.array([[1, 0, 0],
#                      [0, 1, 0]], np.float32)

# 旋转，平移，缩放，切变

# def rotate(mat, angle=180, img_hw=(416, 416), center_yx=(0.5, 0.5)):
#     center_yx = img_hw * center_yx
#     new_mat = cv2.getRotationMatrix2D(center_yx, angle, 1.)
#     mat = mat @ new_mat
#     return mat
#
#
# def move(mat, move_yx=(0.2, 0.2), img_hw=(416, 416), center_yx=(0.5, 0.5)):
#     center_yx = img_hw * center_yx
#     move_yx = img_hw * move_yx
#     cv2.transform()
#
#
# def rotate(mat, angle=180, center_yx=(0.5, 0.5)):
#     pass


# 对坐标应用变换和裁剪



# 对图像应用变换


# 以下代码来自 https://github.com/ultralytics/yolov3/blob/master/utils/datasets.py
def random_affine(img, coords=None, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                  borderValue=(127.5, 127.5, 127.5), inter=cv2.INTER_LINEAR):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    border = 0  # width of added border (optional)
    # height = max(img.shape[0], img.shape[1]) + border * 2
    height, width = img.shape[0] + border * 2, img.shape[1] + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M=M, dsize=(width, height), flags=inter,
                              borderValue=borderValue)  # BGR order borderValue

    affine_params = {
        'M': M,
        'dsize': (width, height),
        'flags': inter,
        'borderValue': borderValue
    }

    # Return warped points also
    if coords is not None:
        coords = np.asarray(coords, np.float32)
        if len(coords) > 0:
            n = coords.shape[0]
            points = coords[:, :4].copy()
            area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # apply angle-based reduction
            radians = a * math.pi / 180
            reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            x = (xy[:, 2] + xy[:, 0]) / 2
            y = (xy[:, 3] + xy[:, 1]) / 2
            w = (xy[:, 2] - xy[:, 0]) * reduction
            h = (xy[:, 3] - xy[:, 1]) * reduction
            xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # reject warped points outside of image
            np.clip(xy, [[0, 0, 0, 0]], [[width, height, width, height]], out=xy)
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

            coords = coords[i]
            coords[:, :4] = xy[i]

        return imw, coords, M
    else:
        return imw, affine_params


def augment_hsv(img):
    # SV augmentation by 20%
    fraction = 0.2
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    S = img_hsv[:, :, 1].astype(np.float32)
    V = img_hsv[:, :, 2].astype(np.float32)

    a = (random.random() * 2 - 1) * fraction + 1
    S *= a
    if a > 1:
        np.clip(S, a_min=0, a_max=255, out=S)

    a = (random.random() * 2 - 1) * fraction + 1
    V *= a
    if a > 1:
        np.clip(V, a_min=0, a_max=255, out=V)

    img_hsv[:, :, 1] = S.astype(np.uint8)
    img_hsv[:, :, 2] = V.astype(np.uint8)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB, dst=img)
