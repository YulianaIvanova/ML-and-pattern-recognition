import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import det, inv
import math
np.random.seed(12)

def check_B(B):
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            if np.abs(B[i,j]) > np.sqrt(B[i,i]*B[j,j]):
                return False
    return True


def gen_norms(M, B):
    A = np.zeros((2, 2), float)
    A[0, 0] = np.sqrt(B[0, 0])
    A[1, 0] = B[0, 1] / np.sqrt(B[0, 0])
    A[1, 1] = np.sqrt(B[1, 1] - pow(B[0, 1], 2) / B[0, 0])

    E = np.random.normal(0,1,2)

    X = A @ E.T + M
    return X


def gen_sample(M, B):
    sample = np.zeros((100,2), float)
    for i in range(100):
        x1 = gen_norms(M, B)
        sample[i, 0:2] = x1
    return sample


#var 8
M1 = np.array([0, 1])
M2 = np.array([1, -1])
M3 = np.array([-1, -1])

# B1 = np.array(
#     [[0.08, 0.03],
#      [0.03, 0.08]])
# # dodgerblue
# B2 = np.array(
#     [[0.08, 0.001],
#      [0.001, 0.08]])

B1 = np.array(
    [[0.4, 0.03],
     [0.03, 0.4]])
# dodgerblue
B2 = np.array(
    [[0.2, 0.01],
     [0.01, 0.2]])
# # firebrick
# B3 = np.array(
#     [[0.08, 0.03],
#      [0.03, 0.08]])


# firebrick
B3 = np.array(
    [[0.08, 0.03],
     [0.03, 0.08]])

#проверка корреляционной матрицы
if check_B(B1) and check_B(B2) and check_B(B3):
    print("Корреляционные матрицы верны!")
else:
    exit("Корреляционные матрицы НЕ верны!")



# sample1 = gen_sample(M1,B1)
# sample2 = gen_sample(M2,B2)

# np.save("sample1_1.npy", sample1, allow_pickle=True)
# np.save("sample1_2.npy", sample2, allow_pickle=True)
#
# plt.scatter(sample1[:,0], sample1[:,1], color='blue', marker='.')
# plt.scatter(sample2[:,0], sample2[:,1], color='red', marker='.')
# plt.grid(True)
# plt.show()
# plt.close()


sample1 = gen_sample(M1,B1)
sample2 = gen_sample(M2,B2)

np.save("sample2_1.npy", sample1, allow_pickle=True)
np.save("sample2_2.npy", sample2, allow_pickle=True)

plt.scatter(sample1[:,0], sample1[:,1], color='blue', marker='.')
plt.scatter(sample2[:,0], sample2[:,1], color='red', marker='.')
plt.grid(True)
plt.show()
plt.close()