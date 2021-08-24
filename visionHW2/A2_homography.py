import cv2
import numpy as np
import time


def img_resize(img, height, width):
    rate_h = height / img.shape[0]
    rate_w = width / img.shape[1]
    result = np.zeros((height, width), dtype=np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            a = int(i//rate_h)
            b = int(j//rate_w)
            result[i, j] = img[a, b]

    return result


def hamming_distance(a, b):
    sz = len(a)
    result = np.zeros(sz, dtype=np.uint8)
    for i in range(0, sz):
        result[i] = bin(a[i]^b[i]).count('1')
    return result


def compute_homography(srcP, destP):
    srcP = np.array(srcP)
    destP = np.array(destP)
    sz = srcP.shape[0]

    meanS = srcP.mean(0)
    meanD = destP.mean(0)
    maxV = 0
    for i in srcP:
        i = i - meanS
        if maxV < i[0]**2 + i[1]**2:
            maxV = i[0]**2 + i[1]**2

    k = 1 / np.sqrt(maxV / 2)
    matS = np.dot([[k, 0, 0], [0, k, 0], [0, 0, 1]],
                  [[1, 0, -meanS[0]], [0, 1, -meanS[1]], [0, 0, 1]])
    srcP = np.insert(srcP, 2, 1, axis=1)
    srcP = np.dot(matS, srcP.T)
    srcP = np.delete(srcP.T, 2, axis=1)

    maxV = 0
    for i in destP:
        i = i - meanD
        if maxV < i[0] ** 2 + i[1] ** 2:
            maxV = i[0] ** 2 + i[1] ** 2

    k = 1 / np.sqrt(maxV / 2)
    matD = np.dot([[k, 0, 0], [0, k, 0], [0, 0, 1]],
                  [[1, 0, -meanD[0]], [0, 1, -meanD[1]], [0, 0, 1]])
    destP = np.insert(destP, 2, 1, axis=1)
    destP = np.dot(matD, destP.T)
    destP = np.delete(destP.T, 2, axis=1)

    M = []
    for i in range(0, sz):
        M.append([-srcP[i][0], -srcP[i][1], -1, 0, 0, 0, srcP[i][0]*destP[i][0], srcP[i][1]*destP[i][0], destP[i][0]])
        M.append([0, 0, 0, -srcP[i][0], -srcP[i][1], -1, srcP[i][0]*destP[i][1], srcP[i][1]*destP[i][1], destP[i][1]])
    M = np.array(M)
    U, S, V = np.linalg.svd(M, full_matrices=True)

    Z = np.reshape(V[8], (3, 3))
    target = np.dot(Z, matS)
    target = np.dot(np.linalg.inv(matD), target)
    target = target / target[2, 2]
    target = np.float32(target)

    return target


def img_compose(img1, img2):
    sh0, sh1 = img1.shape
    result = np.zeros((sh0, sh1), dtype=np.uint8)
    for i in range(0, sh0):
        for j in range(0, sh1):
            if img1[i, j] == 0:
                result[i, j] = img2[i, j]
            else:
                result[i, j] = img1[i, j]

    return result


def img_stitch(img1, img2):
    s1, s2 = img1.shape
    h, w = img2.shape
    result = np.zeros((h, w), dtype=np.uint8)
    result[:, :] = img2[:, :]
    result[0:s1, 0:s2] = img1

    return result


def img_stitch_blending(img1, img2):
    s1, s2 = img1.shape
    h, w = img2.shape
    result = np.zeros((h, w), dtype=np.uint8)
    result[0:s1, 0:s2 - 150] = img1[0:s1, 0:s2 - 150]
    for i in range (0, s1):
        for j in range (s2-150, s2):
            k = (j-(s2-150)) / 150
            result[i, j] = img1[i, j] * (1-k) + img2[i, j] * k
    result[0:s1, s2: w] = img2[0:s1, s2:w]

    return result


def compute_homography_ransac(srcP, destP, th):
    sz = np.shape(srcP)[0]
    start = time.time()
    t = 0

    while time.time() - start < 3:
        pt = np.random.randint(0, sz, (16, ))
        arr1 = []
        arr2 = []
        for i in range(0, 16):
            arr1.append(srcP[pt[i]])
            arr2.append(destP[pt[i]])
        homoM = compute_homography(arr1, arr2)
        count = 0

        for i in range(0, sz):
            p = [[srcP[i][0]], [srcP[i][1]], [1]]
            p = np.dot(homoM, p)
            p = p / p[2][0]
            p = np.delete(p.T, 2, axis=1)
            if np.sqrt((p[0][0]-destP[i][0])**2 + (p[0][1]-destP[i][1])**2) < th:
                count += 1

        if count > t:
            t = count
            target = homoM

    return target


desk = cv2.imread('CV_Assignment_2_Images/cv_desk.png', cv2.IMREAD_GRAYSCALE)
cover = cv2.imread('CV_Assignment_2_Images/cv_cover.jpg', cv2.IMREAD_GRAYSCALE)
cover2 = cv2.imread('CV_Assignment_2_Images/hp_cover.jpg', cv2.IMREAD_GRAYSCALE)
cover2 = img_resize(cover2, cover.shape[0], cover.shape[1])

orb = cv2.ORB_create(nfeatures=900)
kp1 = orb.detect(desk, None)
kp2 = orb.detect(cover, None)
kp1, des1 = orb.compute(desk, kp1)
kp2, des2 = orb.compute(cover, kp2)

des1 = np.array(des1)
des2 = np.array(des2)

temp = []
for i in range(0, len(kp1)):
    rlwns = 10000000
    for j in range(0, len(kp2)):
        k = hamming_distance(des1[i], des2[j]).sum()
        k = np.float(k)
        if k < rlwns:
            rlwns = k
            val = cv2.DMatch(i, j, j, k)
            # queryidx, trainidx, imgidx, distance
            # cover: kp2, desk: kp1, none, k
    temp.append(val)

temp.sort(key=lambda x: x.distance)
made = cv2.drawMatches(desk, kp1, cover, kp2, temp[:10], None, flags=2)
cv2.imshow('made', made)
cv2.waitKey(0)
cv2.destroyAllWindows()


kp1_top = []
kp2_top = []
for i in temp[0:16]:
    kp1_top.append(kp1[i.queryIdx].pt)
    kp2_top.append(kp2[i.trainIdx].pt)

kp1_ransac = []
kp2_ransac = []
for i in temp[0:100]:
    kp1_ransac.append(kp1[i.queryIdx].pt)
    kp2_ransac.append(kp2[i.trainIdx].pt)


homoM = compute_homography(kp2_top, kp1_top)
warp = cv2.warpPerspective(cover, homoM, (desk.shape[1], desk.shape[0]))
cv2.imshow('warp', warp)
cv2.imshow('image_warp', img_compose(warp, desk))

ransac_homoM = compute_homography_ransac(kp2_ransac, kp1_ransac, 3)
ransac_warp = cv2.warpPerspective(cover, ransac_homoM, (desk.shape[1], desk.shape[0]))
cv2.imshow('image_warp_with_ransac', img_compose(ransac_warp, desk))
cv2.imshow('ransac_warp', ransac_warp)
cv2.waitKey(0)
cv2.destroyAllWindows()


warp_harrypotter = cv2.warpPerspective(cover2, ransac_homoM, (desk.shape[1], desk.shape[0]))
cv2.imshow('harrypotter_ransac', warp_harrypotter)
cv2.imshow('warp_harrypotter', img_compose(warp_harrypotter, desk))
cv2.waitKey(0)
cv2.destroyAllWindows()


mountain1 = cv2.imread('CV_Assignment_2_Images/diamondhead-10.png', cv2.IMREAD_GRAYSCALE)
mountain2 = cv2.imread('CV_Assignment_2_Images/diamondhead-11.png', cv2.IMREAD_GRAYSCALE)
orb = cv2.ORB_create(nfeatures=500)
kp1 = orb.detect(mountain1, None)
kp2 = orb.detect(mountain2, None)
kp1, des1 = orb.compute(mountain1, kp1)
kp2, des2 = orb.compute(mountain2, kp2)

des1 = np.array(des1)
des2 = np.array(des2)

temp = []
for i in range(0, len(kp1)):
    rlwns = 100000000
    for j in range(0, len(kp2)):
        k = hamming_distance(des1[i], des2[j]).sum()
        k = np.float(k)
        if k < rlwns:
            rlwns = k
            val = cv2.DMatch(i, j, j, k)
        #queryidx, trainidx, imgidx, distance
        #d-11: kp2, d-10: kp1, none, k
    temp.append(val)

temp.sort(key=lambda x: x.distance)

kp1_ransac = []
kp2_ransac = []
for i in temp[0:100]:
    kp1_ransac.append(kp1[i.queryIdx].pt)
    kp2_ransac.append(kp2[i.trainIdx].pt)

ransac_homoM = compute_homography_ransac(kp2_ransac, kp1_ransac, 1)
shape = mountain1.shape + mountain2.shape
ransac_warp = cv2.warpPerspective(mountain2, ransac_homoM, (mountain1.shape[1]*2, mountain1.shape[0]))
mountain_stitich = img_stitch(mountain1, ransac_warp)
cv2.imshow('image_stitch', mountain_stitich)

mountain_blending = img_stitch_blending(mountain1, ransac_warp)
cv2.imshow('image_stitch_with_blending', mountain_blending)
cv2.waitKey(0)
cv2.destroyAllWindows()