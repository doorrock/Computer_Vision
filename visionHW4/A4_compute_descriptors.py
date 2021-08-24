import cv2
import numpy as np

box = []
center = []
for i in range(100000, 101000):
    f = open("sift/sift" + str(i), "rb")
    a = f.read()
    temp = np.frombuffer(a, dtype=np.uint8)
    temp = temp.reshape(-1, 128)
    box.append(temp)
    center.append(np.average(temp, axis=0))
    f.close()
center = np.array(center, dtype=np.float32)

epoch = 4
N = 1000
D = 1000
for k in range(0, epoch):
    where = np.zeros((N, 1000000), dtype=np.int32)
    new_center = [[] for _ in range(D)]
    for i in range(0, N):
        r = box[i].shape[0]
        for j in range(0, r):
            dist = box[i][j, :] - center
            minIndex = np.argmin(np.linalg.norm(dist, axis=1))
            where[i, j] = minIndex
            # which center is nearest
            new_center[minIndex].append(box[i][j, :])


    # test
    # result0 = np.zeros((N, D), dtype=np.float32)
    # result1 = np.zeros((N, D), dtype=np.float32)
    # result2 = np.zeros((N, D), dtype=np.float32)
    # result3 = np.zeros((N, D), dtype=np.float32)
    # result4 = np.zeros((N, D), dtype=np.float32)
    # result5 = np.zeros((N, D), dtype=np.float32)
    # for i in range(0, N):
    #     r = box[i].shape[0]
    #     for j in range(0, r):
    #         result0[i, where[i, j]] += 1
    #         result1[i, where[i, j]] += np.power(np.linalg.norm(box[i][j] - center[where[i, j]]), 1) / r
    #         result2[i, where[i, j]] += np.power(np.linalg.norm(box[i][j] - center[where[i, j]]), 1 / 2) / r
    #         result3[i, where[i, j]] += np.power(np.linalg.norm(box[i][j] - center[where[i, j]]), 1 / 3) / r
    #         result4[i, where[i, j]] += np.power(np.linalg.norm(box[i][j] - center[where[i, j]]), 1 / 4) / r
    #         result5[i, where[i, j]] += np.power(np.linalg.norm(box[i][j] - center[where[i, j]]), 1 / 5) / r
    if k + 1 == epoch:
        result = np.zeros((N, D), dtype=np.float32)
        for i in range(0, N):
            r = box[i].shape[0]
            for j in range(0, r):
                result[i, where[i, j]] += np.power(np.linalg.norm(box[i][j] - center[where[i, j]]), 1 / 5) / r

    # calculate new center
    for i in range(0, D):
        temp = np.array(new_center[i])
        sumV = np.sum(temp, axis=0)
        sumV = sumV / (temp.shape[0] + 0.000001)
        center[i][:] = sumV


    # ff0 = open("A4_2016310493" + "_epoch_0" + str(k) + ".des", "wb")
    # ff1 = open("A4_2016310493" + "_epoch_1" + str(k) + ".des", "wb")
    # ff2 = open("A4_2016310493" + "_epoch_2" + str(k) + ".des", "wb")
    # ff3 = open("A4_2016310493" + "_epoch_3" + str(k) + ".des", "wb")
    # ff4 = open("A4_2016310493" + "_epoch_4" + str(k) + ".des", "wb")
    # ff5 = open("A4_2016310493" + "_epoch_5" + str(k) + ".des", "wb")

    # NN = np.array([N])
    # DD = np.array([D])
    # ff0.write(NN.tobytes())
    # ff0.write(DD.tobytes())
    # ff1.write(NN.tobytes())
    # ff1.write(DD.tobytes())
    # ff2.write(NN.tobytes())
    # ff2.write(DD.tobytes())
    # ff3.write(NN.tobytes())
    # ff3.write(DD.tobytes())
    # ff4.write(NN.tobytes())
    # ff4.write(DD.tobytes())
    # ff5.write(NN.tobytes())
    # ff5.write(DD.tobytes())

    # for i in range(0, N):
    #     for j in range(0, D):
    #         ff0.write(result0[i, j].tobytes())
    #         ff1.write(result1[i, j].tobytes())
    #         ff2.write(result2[i, j].tobytes())
    #         ff3.write(result3[i, j].tobytes())
    #         ff4.write(result4[i, j].tobytes())
    #         ff5.write(result5[i, j].tobytes())
    # ff0.close()
    # ff1.close()
    # ff2.close()
    # ff3.close()
    # ff4.close()
    # ff5.close()
ff = open("A4_2016310493.des", "wb")
NN = np.array([N])
DD = np.array([D])
ff.write(NN.tobytes())
ff.write(DD.tobytes())
for i in range(0, N):
    for j in range(0, D):
        ff.write(result[i, j].tobytes())
ff.close()