import cv2
import numpy as np
from skimage.exposure import rescale_intensity
import math

sobel_x = [[1, 0, -1],
           [2, 0, -2],
           [1, 0, -1]]
sobel_y = [[1, 2, 1],
           [0, 0, 0],
           [-1, -2, -1]]
sobel_x = np.array(sobel_x)
sobel_y = np.array(sobel_y)


def textONimage(image, text):
    cv2.putText(image, text, (30, 50), 0, 1, (0, 0, 0), 2)


def padding(image, val_x, val_y, mode):
    x = image.shape[0] + val_x * 2
    y = image.shape[1] + val_y * 2
    result = np.zeros((x, y), dtype=np.float32)
    top = 0
    bottom = image.shape[0] - 1;
    left = 0
    right = image.shape[1] - 1;
    if mode != 0:
        for k in range (0, val_x):
            result[k, val_y: y - val_y] = image[top, 0: y - val_y]
        for k in range(x - val_x, x):
            result[k, val_y: y - val_y] = image[bottom, 0: y - val_y]
        for k in range(0, val_y):
            result[val_x: x - val_x, k] = image[0: x - val_x, left]
        for k in range(y - val_y, y):
            result[val_x: x - val_x, k] = image[0: x - val_x, right]
        result[0: val_x, 0: val_y] = image[top, left]
        result[x - val_x: x, 0: val_y] = image[bottom, left]
        result[0: val_x, y - val_y: y] = image[top, right]
        result[x - val_x: x, y - val_y: y] = image[bottom, right]
    result[val_x: x - val_x, val_y: y - val_y] = image[top:bottom + 1, left:right + 1]

    if mode != 0:
        result.astype(np.uint8)
    return result


def get_gaussian_filter_1d(size, sigma):
    array = np.zeros((size, ), dtype=np.float32)
    st = size // 2
    for i in range(-st, st + 1):
        array[i + st] = math.exp((-(i*i)/(2*(sigma*sigma))))
    s = array.sum()
    array = array / s
    return array


def get_gaussian_filter_2d(size, sigma):
    arr = get_gaussian_filter_1d(size, sigma)
    arr = np.array([arr])
    array = np.dot(arr.T, arr)
    return array


def cross_correlation_1d(image, kernel):
    x = image.shape[0]
    y = image.shape[1]
    horizontal = True
    amount = kernel.shape[0] // 2
    output = np.zeros((x, y), dtype=np.float32)
    d = kernel.ndim
    if d == 2:
        horizontal = False

    if horizontal:
        padding_image = padding(image, 0, amount, 1)
        for i in range(0, x):
            for j in range(amount, y + amount):
                part = padding_image[i, j - amount : j + amount + 1]
                t = (part * kernel).sum()
                output[i, j - amount] = t
    else:
        padding_image = padding(image, amount, 0, 1)
        for i in range(amount, x + amount):
            for j in range(0, y):
                part = padding_image[i - amount : i + amount + 1, j]
                t = (part * kernel.T).sum()
                output[i - amount, j] = t

    output = rescale_intensity(output, (0, 255), (0, 255)).astype(np.uint8)

    return output


def cross_correlation_2d(image, kernel):
    x = image.shape[0]
    y = image.shape[1]
    output = np.zeros((x, y), dtype=np.float32)
    amount_x = kernel.shape[0] // 2
    amount_y = kernel.shape[1] // 2
    padding_image = padding(image, amount_x, amount_y, 1)

    for i in range(amount_x, x + amount_x):
        for j in range(amount_y, y + amount_y):
            part = padding_image[i - amount_x:i + amount_x + 1, j - amount_y:j + amount_y + 1]
            t = (part * kernel).sum()
            output[i - amount_x, j - amount_y] = t

    output = rescale_intensity(output, (0, 255), (0, 255)).astype(np.uint8)

    return output


def sobel(image):
    x = image.shape[0]
    y = image.shape[1]
    output_x = np.zeros((x, y), dtype=np.float32)
    output_y = np.zeros((x, y), dtype=np.float32)
    padding_image = padding(image, 1, 1, 1)

    for i in range(0, x):
        for j in range(0, y):
            part = padding_image[i: i + 3, j: j + 3]
            t_x = (part * sobel_x).sum()
            t_y = (part * sobel_y).sum()
            output_x[i, j] = t_x
            output_y[i, j] = t_y

    return output_x, output_y


def compute_image_gradient(image):
    x = image.shape[0]
    y = image.shape[1]
    output_mag = np.zeros((x, y), dtype=np.float32)
    output_dir = np.zeros((x, y), dtype=np.float32)
    padding_image = padding(image, 1, 1, 1)

    for i in range(0, x):
        for j in range(0, y):
            part = padding_image[i: i + 3, j: j + 3]
            xx = (part * sobel_x).sum()
            yy = (part * sobel_y).sum()
            if xx == 0:
                output_dir[i, j] = 90
            else:
                output_dir[i, j] = np.arctan(yy / -xx) * 180 / math.pi
            output_mag[i, j] = math.sqrt(xx**2 + yy**2)

    output_mag = rescale_intensity(output_mag, out_range=(0, 255)).astype(np.uint8)

    return output_mag, output_dir
