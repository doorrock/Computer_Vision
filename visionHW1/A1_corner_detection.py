import cv2
import numpy as np
from skimage.exposure import rescale_intensity
from base import *
import math
import time


def compute_corner_response(image):
    x = image.shape[0]
    y = image.shape[1]

    output = np.zeros((x, y), dtype=np.float32)
    image_sobel_x, image_sobel_y = sobel(image)
    image_sobel_x = padding(image_sobel_x, 2, 2, 0)
    image_sobel_y = padding(image_sobel_y, 2, 2, 0)
    m = np.zeros((2, 2), dtype=np.float32)
    xx = image_sobel_x * image_sobel_x
    xy = image_sobel_x * image_sobel_y
    yy = image_sobel_y * image_sobel_y
    for i in range(2, x + 2):
        for j in range(2, y + 2):
            m00 = xx[i - 2: i + 3, j - 2: j + 3].sum()
            m01 = xy[i - 2: i + 3, j - 2: j + 3].sum()
            m11 = yy[i - 2: i + 3, j - 2: j + 3].sum()
            det = m00 * m11 - m01 * m01
            trace = m00 + m11
            r = det - 0.04 * (trace ** 2)
            if r > 0:
                output[i - 2, j - 2] = r

    output = rescale_intensity(output, out_range=(0, 1))
    return output


def non_maximum_suppression_win(r, winSize):
    x = r.shape[0]
    y = r.shape[1]
    amount = winSize // 2

    output = np.zeros((x, y), dtype=np.float32)
    padding_image = padding(r, amount, amount, 0)
    for i in range(amount, x + amount):
        for j in range(amount, y + amount):
            part = padding_image[i - amount: i + amount + 1, j - amount: j + amount + 1]
            if (np.max(part) == padding_image[i, j]) and (padding_image[i, j] > 0.1):
                output[i - amount, j - amount] = padding_image[i, j]

    return output


def change_color(ori_image, corner_image):
    x = ori_image.shape[0]
    y = ori_image.shape[1]
    output = np.copy(ori_image)
    output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    for i in range(0, x):
        for j in range(0, y):
            if corner_image[i, j] > 0.1:
                output[i, j] = [0, 255, 0]

    return output


def draw_circle(ori_image, corner_image):
    x = ori_image.shape[0]
    y = ori_image.shape[1]
    output = np.copy(ori_image)
    output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    for i in range(0, x):
        for j in range(0, y):
            if corner_image[i, j] > 0.1:
                cv2.circle(output, (j, i), 4, (0, 255, 0), 2)

    return output


if(__name__ == "__main__"):
    gf = get_gaussian_filter_2d(7, 1.5)

    ori_lenna = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
    gau_lenna = cross_correlation_2d(ori_lenna, gf)
    now_time = time.time()
    corner_lenna = compute_corner_response(gau_lenna)
    print("lenna_1: {}".format(time.time() - now_time))
    cv2.imshow("corner_lenna", corner_lenna)
    cv2.waitKey(0)

    ori_shapes = cv2.imread('shapes.png', cv2.IMREAD_GRAYSCALE)
    gau_shapes = cross_correlation_2d(ori_shapes, gf)
    now_time = time.time()
    corner_shapes = compute_corner_response(gau_shapes)
    print("shapes_1: {}".format(time.time() - now_time))
    cv2.imshow("corner_shapes", corner_shapes)
    cv2.waitKey(0)

    green_lenna = change_color(ori_lenna, corner_lenna)
    cv2.imshow('part_3_corner_bin_lenna.png', green_lenna)
    cv2.imwrite('./result/part_3_corner_bin_lenna.png', green_lenna)
    cv2.waitKey(0)

    green_shapes = change_color(ori_shapes, corner_shapes)
    cv2.imshow('part_3_corner_bin_shapes.png', green_shapes)
    cv2.imwrite('./result/part_3_corner_bin_shapes.png', green_shapes)
    cv2.waitKey(0)

    now_time = time.time()
    suppressed_corner_lenna = non_maximum_suppression_win(corner_lenna, 11)
    print("lenna_2: {}".format(time.time() - now_time))
    target = draw_circle(ori_lenna, suppressed_corner_lenna)
    cv2.imshow('part_3_corner_sup_lenna.png', target)
    cv2.imwrite('./result/part_3_corner_sup_lenna.png', target)
    cv2.waitKey(0)

    now_time = time.time()
    suppressed_corner_shapes = non_maximum_suppression_win(corner_shapes, 11)
    print("shapes_2: {}".format(time.time() - now_time))
    target = draw_circle(ori_shapes, suppressed_corner_shapes)
    cv2.imshow('part_3_corner_sup_shapes.png', target)
    cv2.imwrite('./result/part_3_corner_sup_shapes.png', target)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
