import cv2
import numpy as np

def get_transformed_image(img, M):
    x = img.shape[0]
    y = img.shape[1]
    background = np.full((801, 801), 255, dtype=np.uint8)
    cv2.arrowedLine(background, (400, 800), (400, 0), (0, 0, 0), 2, tipLength=0.05)
    cv2.arrowedLine(background, (0, 400), (800, 400), (0, 0, 0), 2, tipLength=0.05)

    xx = int(x/2)
    yy = int(y/2)
    for i in range(0, x):
        for j in range(0, y):
            whkvy = np.dot(M, [[i - x/2], [j - y/2], [1]])
            whkvy = whkvy / whkvy[2, 0]
            whkvy += [[x/2], [y/2], [0]]
            whkvy = np.round(whkvy)
            whkvy = np.int32(whkvy)
            background[400 + whkvy[0][0] - xx, 400 + whkvy[1][0] - yy] = np.bitwise_and(background[400 + whkvy[0][0] - xx, 400 + whkvy[1][0] - yy], img[i, j])

    return background


smile = cv2.imread('CV_Assignment_2_Images/smile.png', cv2.IMREAD_GRAYSCALE)
height = smile.shape[0]//2
width = smile.shape[1]//2
M = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
while 1:
    back = get_transformed_image(smile, M)
    cv2.imshow('background', back)
    key = cv2.waitKey(0)
    if key == 97:
        M[1][2] -= 5
    elif key == 100:
        M[1][2] += 5
    elif key == 119:
        M[0][2] -= 5
    elif key == 115:
        M[0][2] += 5
    elif key == 114:
        rot = np.float32([[np.cos(np.pi/36), -np.sin(np.pi/36), 0], [np.sin(np.pi/36), np.cos(np.pi/36) ,0], [0, 0, 1]])
        M = np.dot(rot, M)
    elif key == 82:
        rot = np.float32([[np.cos(np.pi/36), np.sin(np.pi/36), 0], [-np.sin(np.pi/36), np.cos(np.pi/36), 0], [0, 0, 1]])
        M = np.dot(rot, M)
    elif key == 102:
        flip = np.float32([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        M = np.dot(flip, M)
    elif key == 70:
        flip = np.float32([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        M = np.dot(flip, M)
    elif key == 120:
        size = np.float32([[1, 0, 0], [0, 0.95, 0], [0, 0, 1]])
        M = np.dot(size, M)
    elif key == 88:
        size = np.float32([[1, 0, 0], [0, 1.05, 0], [0, 0, 1]])
        M = np.dot(size, M)
    elif key == 121:
        size = np.float32([[0.95, 0, 0], [0, 1, 0], [0, 0, 1]])
        M = np.dot(size, M)
    elif key == 89:
        size = np.float32([[1.05, 0, 0], [0, 1, 0], [0, 0, 1]])
        M = np.dot(size, M)
    elif key == 72:
        M = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    elif key == 81:
        break

cv2.destroyAllWindows()