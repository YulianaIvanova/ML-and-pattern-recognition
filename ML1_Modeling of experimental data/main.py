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

    X = np.dot(A,E) + M
    return X


def gen_sample(M, B):
    sample = np.zeros((200,2), float)
    for i in range(200):
        x1 = gen_norms(M, B)
        sample[i, 0:2] = x1
    return sample

def calc_M_hat(sample):
    M_hat = np.zeros(2, float)
    M_hat[0] = sum(sample[:, 0]) / sample.shape[0]
    M_hat[1] = sum(sample[:, 1]) / sample.shape[0]
    return M_hat

def calc_B_hat(sample):
    M_hat = calc_M_hat(sample)
    B = np.zeros((2,2))
    n = sample.shape[0]
    for i in range(n):
        B += np.dot(sample[i,:].reshape(2,1), sample[i,:].reshape(1,2))
    B /= n
    B -= np.dot(M_hat.reshape(2,1), M_hat.reshape(1,2))
    #print(B)
    return B

def bhatachari(M1, M0, B1, B0):
    M = M1 - M0
    B = (B1 + B0) / 2

    res = 1/4 * (M.transpose() @ inv(B) @ M) + 1/2 * \
          math.log(det(B)/np.sqrt(det(B1)*det(B0)))
    return res

def mahalanobis(M0, M1, B):
    M = M1 - M0
    res = M.transpose() @ inv(B) @ M
    return res

#var 8
M1 = np.array([0, 1])
M2 = np.array([1, -1])
M3 = np.array([-1, -1])

B1 = np.array([[0.2, 0.1],
               [0.1, 0.2]])

B2 = np.array([[0.3, 0],
               [0, 0.3]])

B3 = np.array([[0.7, -0.2],
               [-0.2, 0.9]])

#проверка корреляционной матрицы
if check_B(B1) and check_B(B2) and check_B(B3):
    print("Корреляционные матрицы верны!")
else:
    exit("Корреляционные матрицы НЕ верны!")



X1 = gen_norms(M1, B1)
X2 = gen_norms(M2, B2)
X3 = gen_norms(M3, B3)
#print(X1)
#print(X2)
#print(X3)

sample1 = gen_sample(M1,B1)
sample2 = gen_sample(M2,B1)
plt.scatter(sample1[:,0], sample1[:,1], color='blue', marker='.')
plt.scatter(sample2[:,0], sample2[:,1], color='red', marker='.')
plt.grid(True)
plt.show()
plt.close()
np.save("sample1_1.npy", sample1, allow_pickle=True)
np.save("sample1_2.npy", sample2, allow_pickle=True)



sample1 = gen_sample(M1,B1)
sample2 = gen_sample(M2,B2)
sample3 = gen_sample(M3,B3)
plt.scatter(sample1[:,0], sample1[:,1], color='blue', marker='.')
plt.scatter(sample2[:,0], sample2[:,1], color='red', marker='.')
plt.scatter(sample3[:,0], sample3[:,1], color='green', marker='.')
plt.grid(True)
plt.show()
plt.close()

np.save("sample2_1.npy", sample1, allow_pickle=True)
np.save("sample2_2.npy", sample2, allow_pickle=True)
np.save("sample2_3.npy", sample3, allow_pickle=True)

print("Точечные оценки")
print(f'Оценка Матожидания 1 выборки: {calc_M_hat(sample1)}')
print(f'Оценка Матожидания 2 выборки: {calc_M_hat(sample2)}')
print(f'Оценка Матожидания 3 выборки: {calc_M_hat(sample3)} \n')

print(f'Оценка корреляционной матрицы 1 выборки: \n{calc_B_hat(sample1)}\n')
print(f'Оценка корреляционной матрицы 2 выборки: \n{calc_B_hat(sample2)}\n')
print(f'Оценка корреляционной матрицы 3 выборки: \n{calc_B_hat(sample3)}\n')

print(f"Bhatachari(1-2): {bhatachari(M1, M2, B1, B2):.6f}")
print(f"Bhatachari(1-3): {bhatachari(M1, M3, B1, B3):.6f}")
print(f"Bhatachari(3-2): {bhatachari(M3, M2, B3, B2):.6f}")
print(f"Mahalanobis(1-2): {mahalanobis(M1, M2, B1):.6f}")
