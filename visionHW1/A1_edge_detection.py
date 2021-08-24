import cv2
import numpy as np
from skimage.exposure import rescale_intensity
from base import *
import math
import time


def closet(val):
    if -90 <= val < -67.5:
        return 1, 0, -1, 0
    elif -67.5 <= val < -22.5:
        return -1, -1, 1, 1
    elif -22.5 <= val < 22.5:
        return 0, 1, 0, -1
    elif 22.5 <= val < 67.5:
        return -1, 1, 1, -1
    elif 67.5 <= val <= 90:
        return 1, 0, -1, 0


def non_maximum_suppression_dir(mag, dir):
    x = mag.shape[0]
    y = mag.shape[1]
    output = np.zeros((x, y), dtype=np.float32)
    mag = padding(mag, 1, 1, 0)
    for i in range(1, x + 1):
        for j in range(1, y + 1):
            x1, y1, x2, y2 = closet(dir[i - 1, j - 1])
            k = max(mag[i + x1, j + y1], mag[i + x2, j + y2], mag[i, j])
            if k == mag[i, j]:
                output[i - 1, j - 1] = mag[i, j]

    output = rescale_intensity(output, out_range=(0, 255)).astype(np.uint8)
    return output


if(__name__ == "__main__"):
    gf = get_gaussian_filter_2d(7, 1.5)

    img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
    img = cross_correlation_2d(img, gf)
    now_time = time.time()
    output_lenna, dir_lenna = compute_image_gradient(img)
    print("lenna_1: {}".format(time.time() - now_time))
    cv2.imwrite('./result/part_2_edge_lenna.png', output_lenna)
    cv2.imshow('lenna_edge', output_lenna)
    cv2.waitKey(0)

    img = cv2.imread('shapes.png', cv2.IMREAD_GRAYSCALE)
    img = cross_correlation_2d(img, gf)
    now_time = time.time()
    output_shapes, dir_shapes = compute_image_gradient(img)
    print("shapes_1: {}".format(time.time() - now_time))
    cv2.imwrite('./result/part_2_edge_shapes.png', output_shapes)
    cv2.imshow('shapes_edge', output_shapes)
    cv2.waitKey(0)

    now_time = time.time()
    NMS_lenna = non_maximum_suppression_dir(output_lenna, dir_lenna)
    print("lenna_2: {}".format(time.time() - now_time))
    cv2.imshow('NMS_lenna', NMS_lenna)
    cv2.imwrite('./result/part_2_edge_sup_lenna.png', NMS_lenna)
    cv2.waitKey(0)

    now_time = time.time()
    NMS_shapes = non_maximum_suppression_dir(output_shapes, dir_shapes)
    print("shapes_2: {}".format(time.time() - now_time))
    cv2.imshow('NMS_shapes', NMS_shapes)
    cv2.imwrite('./result/part_2_edge_sup_shapes.png', NMS_shapes)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
