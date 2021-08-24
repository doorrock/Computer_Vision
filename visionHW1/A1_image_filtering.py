import cv2
import numpy as np
from skimage.exposure import rescale_intensity
from base import *
import math
import time


def show(image):
    gf = get_gaussian_filter_2d(5, 1)
    filtered_img_1 = cross_correlation_2d(image, gf)
    textONimage(filtered_img_1, '5x5 s=1')
    gf = get_gaussian_filter_2d(11, 1)
    filtered_img_2 = cross_correlation_2d(image, gf)
    textONimage(filtered_img_2, '11x11 s=1')
    gf = get_gaussian_filter_2d(17, 1)
    filtered_img_3 = cross_correlation_2d(image, gf)
    textONimage(filtered_img_3, '17x17 s=1')

    making = cv2.vconcat([filtered_img_1, filtered_img_2])
    making = cv2.vconcat([making, filtered_img_3])
    target = making

    gf = get_gaussian_filter_2d(5, 6)
    filtered_img_1 = cross_correlation_2d(image, gf)
    textONimage(filtered_img_1, '5x5 s=6')
    gf = get_gaussian_filter_2d(11, 6)
    filtered_img_2 = cross_correlation_2d(image, gf)
    textONimage(filtered_img_2, '11x11 s=6')
    gf = get_gaussian_filter_2d(17, 6)
    filtered_img_3 = cross_correlation_2d(image, gf)
    textONimage(filtered_img_3, '17x17 s=6')

    making = cv2.vconcat([filtered_img_1, filtered_img_2])
    making = cv2.vconcat([making, filtered_img_3])
    target = cv2.hconcat([target, making])

    gf = get_gaussian_filter_2d(5, 11)
    filtered_img_1 = cross_correlation_2d(image, gf)
    textONimage(filtered_img_1, '5x5 s=11')
    gf = get_gaussian_filter_2d(11, 11)
    filtered_img_2 = cross_correlation_2d(image, gf)
    textONimage(filtered_img_2, '11x11 s=11')
    gf = get_gaussian_filter_2d(17, 11)
    filtered_img_3 = cross_correlation_2d(image, gf)
    textONimage(filtered_img_3, '17x17 s=11')

    making = cv2.vconcat([filtered_img_1, filtered_img_2])
    making = cv2.vconcat([making, filtered_img_3])
    target = cv2.hconcat([target, making])

    return target


if(__name__ == "__main__"):

    print("gaussian filter 1d : \n{}".format(get_gaussian_filter_1d(5, 1)))
    print("gaussian filter 2d : \n{}".format(get_gaussian_filter_2d(5, 1)))

    img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
    target = show(img)
    cv2.imwrite("./result/part_1_gaussian_filtered_lenna.png", target)
    cv2.imshow('part_1_gaussian_filtered_lenna.png', target)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img = cv2.imread('shapes.png', cv2.IMREAD_GRAYSCALE)
    target = show(img)
    cv2.imwrite("./result/part_1_gaussian_filtered_shapes.png", target)
    cv2.imshow('part_1_gaussian_filtered_shapes.png', target)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    gf1d_17_6 = get_gaussian_filter_1d(17, 6)
    gf1d_17_6_T = np.array([gf1d_17_6]).T
    gf2d_17_6 = get_gaussian_filter_2d(17, 6)

    img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
    now_time = time.time()
    out1d = cross_correlation_1d(img, gf1d_17_6_T)
    out1d = cross_correlation_1d(out1d, gf1d_17_6)
    print("1d_lenna: {}".format(time.time()-now_time))
    now_time = time.time()
    out2d = cross_correlation_2d(img, gf2d_17_6)
    print("2d_lenna: {}".format(time.time()-now_time))
    cv2.waitKey(0)

    diff = cv2.absdiff(out1d, out2d)
    cv2.imshow('diff_lenna', diff)
    print("Sum of diff_lenna: {}".format(diff.sum()))
    cv2.waitKey(0)

    img = cv2.imread('shapes.png', cv2.IMREAD_GRAYSCALE)
    now_time = time.time()
    out1d = cross_correlation_1d(img, gf1d_17_6_T)
    out1d = cross_correlation_1d(out1d, gf1d_17_6)
    print("1d_shapes: {}".format(time.time()-now_time))
    now_time = time.time()
    out2d = cross_correlation_2d(img, gf2d_17_6)
    print("2d_shapes: {}".format(time.time()-now_time))

    diff = cv2.absdiff(out1d, out2d)
    cv2.imshow('diff_shapes', diff)
    print("Sum of diff_shapes: {}".format(diff.sum()))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
