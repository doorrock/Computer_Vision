import cv2
import numpy as np
import time
import compute_avg_reproj_error
shape = ()

def compute_F_raw(M):
    x, y = M.shape
    T = []
    for i in range(0, x):
        MM = M[i]
        T.append([MM[0]*MM[2], MM[0]*MM[3], MM[0], MM[1]*MM[2], MM[1]*MM[3], MM[1], MM[2], MM[3], 1])
    T = np.array(T)
    U, S, V = np.linalg.svd(T, full_matrices=True)

    Z = np.reshape(V[8], (3, 3))
    return Z


def compute_F_norm(M):
    row, col, zz = shape
    row /= 2
    col /= 2
    P1 = M[:, 0:2]
    P2 = M[:, 2:4]
    P1 = np.insert(P1, 2, 1, axis=1)
    P2 = np.insert(P2, 2, 1, axis=1)

    normalize = np.dot([[1 / col, 0, 0], [0, 1 / row, 0], [0, 0, 1]], [[1, 0, -col], [0, 1, -row], [0, 0, 1]])
    P1 = np.dot(normalize, P1.T)
    P2 = np.dot(normalize, P2.T)
    T = []
    for i in range(0, M.shape[0]):
        T.append([P1[0, i] * P2[0, i], P1[0, i] * P2[1, i], P1[0, i], P1[1, i] * P2[0, i], P1[1, i] * P2[1, i], P1[1, i], P2[0, i], P2[1, i], 1])
    T = np.array(T)
    U, S, V = np.linalg.svd(T, full_matrices=True)

    Z = np.reshape(V[8], (3, 3))
    U, S, V = np.linalg.svd(Z, full_matrices=True)
    S = [[S[0], 0, 0], [0, S[1], 0], [0, 0, 0]]
    Z = np.dot(S, V)
    Z = np.dot(U, Z)
    Z = np.dot(Z, normalize)
    Z = np.dot(normalize.T, Z)
    return Z


def compute_F_mine(M):
    sz = M.shape[0]
    P1 = M[:, 0:2]
    P2 = M[:, 2:4]
    P1 = np.insert(P1, 2, 1, axis=1)
    P2 = np.insert(P2, 2, 1, axis=1)

    start = time.time()
    rlwns = 0
    while time.time() - start < 3:
        ran = np.random.randint(0, sz, (24, ))
        A = []
        for i in ran:
            A.append(M[i])
        A = np.array(A)
        Z = compute_F_norm(A)

        count = 0

        L = np.dot(Z, P1.T)
        val = np.dot(L.T, P2.T)
        for i in range(0, sz):
            d = (val[i, i])**2 / (L[0, i]**2 + L[1, i]**2)
            if d < 3:
                count = count + 1
        if count > rlwns:
            rlwns = count
            target = Z.T

    return target


def drawlines(M, F, img1, img2):
    for cnt in range(0, 3):
        t1 = np.copy(img1)
        t2 = np.copy(img2)
        ran = np.random.randint(0, M.shape[0], (3,))
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

        t1 = cv2.circle(t1, tuple((int(M[ran[0], 0]), int(M[ran[0], 1]))), 5, color[0], -1)
        t2 = cv2.circle(t2, tuple((int(M[ran[0], 2]), int(M[ran[0], 3]))), 5, color[0], -1)
        t1 = cv2.circle(t1, tuple((int(M[ran[1], 0]), int(M[ran[1], 1]))), 5, color[1], -1)
        t2 = cv2.circle(t2, tuple((int(M[ran[1], 2]), int(M[ran[1], 3]))), 5, color[1], -1)
        t1 = cv2.circle(t1, tuple((int(M[ran[2], 0]), int(M[ran[2], 1]))), 5, color[2], -1)
        t2 = cv2.circle(t2, tuple((int(M[ran[2], 2]), int(M[ran[2], 3]))), 5, color[2], -1)

        w = np.dot(F.T, [[M[ran[0], 2]], [M[ran[0], 3]], [1]])
        x0, y0 = map(int, [0, -w[2] / w[1]])
        x1, y1 = map(int, [shape[1], -(w[2] + w[0] * shape[1]) / w[1]])
        t1 = cv2.line(t1, (x0, y0), (x1, y1), color[0], 1)

        w = np.dot(F.T, [[M[ran[1], 2]], [M[ran[1], 3]], [1]])
        x0, y0 = map(int, [0, -w[2] / w[1]])
        x1, y1 = map(int, [shape[1], -(w[2] + w[0] * shape[1]) / w[1]])
        t1 = cv2.line(t1, (x0, y0), (x1, y1), color[1], 1)

        w = np.dot(F.T, [[M[ran[2], 2]], [M[ran[2], 3]], [1]])
        x0, y0 = map(int, [0, -w[2] / w[1]])
        x1, y1 = map(int, [shape[1], -(w[2] + w[0] * shape[1]) / w[1]])
        t1 = cv2.line(t1, (x0, y0), (x1, y1), color[2], 1)

        w = np.dot(F, [[M[ran[0], 0]], [M[ran[0], 1]], [1]])
        x0, y0 = map(int, [0, -w[2] / w[1]])
        x1, y1 = map(int, [shape[1], -(w[2] + w[0] * shape[1]) / w[1]])
        t2 = cv2.line(t2, (x0, y0), (x1, y1), color[0], 1)

        w = np.dot(F, [[M[ran[1], 0]], [M[ran[1], 1]], [1]])
        x0, y0 = map(int, [0, -w[2] / w[1]])
        x1, y1 = map(int, [shape[1], -(w[2] + w[0] * shape[1]) / w[1]])
        t2 = cv2.line(t2, (x0, y0), (x1, y1), color[1], 1)

        w = np.dot(F, [[M[ran[2], 0]], [M[ran[2], 1]], [1]])
        x0, y0 = map(int, [0, -w[2] / w[1]])
        x1, y1 = map(int, [shape[1], -(w[2] + w[0] * shape[1]) / w[1]])
        t2 = cv2.line(t2, (x0, y0), (x1, y1), color[2], 1)

        cv2.imshow('1', t1)
        cv2.imshow('2', t2)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if key == 113:
            break


temple1 = cv2.imread('temple1.png', cv2.IMREAD_COLOR)
temple2 = cv2.imread('temple2.png', cv2.IMREAD_COLOR)
M = np.loadtxt('temple_matches.txt')
shape = temple1.shape
rawF = compute_F_raw(M)
normF = compute_F_norm(M)
mineF = compute_F_mine(M)
print("Average Reprojection Errors (temple1.png and temple2.png)")
print("\tRaw =", compute_avg_reproj_error.compute_avg_reproj_error(M, rawF))
print("\tNorm =", compute_avg_reproj_error.compute_avg_reproj_error(M, normF))
print("\tMine =", compute_avg_reproj_error.compute_avg_reproj_error(M, mineF))
drawlines(M, mineF, temple1, temple2)

house1 = cv2.imread('house1.jpg', cv2.IMREAD_COLOR)
house2 = cv2.imread('house2.jpg', cv2.IMREAD_COLOR)
M = np.loadtxt('house_matches.txt')
shape = house1.shape
rawF = compute_F_raw(M)
normF = compute_F_norm(M)
mineF = compute_F_mine(M)
print("Average Reprojection Errors (house1.jpg and house2.jpg)")
print("\tRaw =", compute_avg_reproj_error.compute_avg_reproj_error(M, rawF))
print("\tNorm =", compute_avg_reproj_error.compute_avg_reproj_error(M, normF))
print("\tMine =", compute_avg_reproj_error.compute_avg_reproj_error(M, mineF))
drawlines(M, mineF, house1, house2)


library1 = cv2.imread('library1.jpg', cv2.IMREAD_COLOR)
library2 = cv2.imread('library2.jpg', cv2.IMREAD_COLOR)
M = np.loadtxt('library_matches.txt')
shape = library1.shape
rawF = compute_F_raw(M)
normF = compute_F_norm(M)
mineF = compute_F_mine(M)
print("Average Reprojection Errors (library1.jpg and library2.jpg)")
print("\tRaw =", compute_avg_reproj_error.compute_avg_reproj_error(M, rawF))
print("\tNorm =", compute_avg_reproj_error.compute_avg_reproj_error(M, normF))
print("\tMine =", compute_avg_reproj_error.compute_avg_reproj_error(M, mineF))
drawlines(M, mineF, library1, library2)