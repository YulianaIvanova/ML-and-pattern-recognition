import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, norm
from random import randint
from numpy import sign as sgn


def shuffle(a: list):
    """Возвращает перемешанную копию списка"""
    b = a.copy()
    n = len(a)
    for i in range(n-1):
        r = randint(i+1, n-1)
        b[i], b[r] = b[r], b[i]
    return b



def fisher_classifier1(B, M1, M2):
    W = inv(B) @ (M2 - M1)
    wn = -0.5 * np.transpose(M2 - M1) @ inv(B) @ (M2 + M1)
    return W, wn


def fisher_classifier2(B1, B2, M1, M2):
    W = 2 * inv(B1 + B2) @ (M2 - M1)
    o1 = W.T @ B1 @ W
    o2 = W.T @ B2 @ W
    wn = - ((M2 - M1).T @ (2 * inv(B1 + B2)) @ (o2 * M1 + o1 * M2)) / (o1 + o2)
    return W, wn


def minstd_classifier(sample1, sample2):
    np_one = np.ones(sample1.shape[1])
    z1 = np.vstack([-1*sample1, -1*np_one])
    z2 = np.vstack([sample2, np_one])

    U = np.ones((len(z1), len(z1[0]) + len(z2[0])))

    U[:, :len(z1[0])] = z2.copy()
    U[:, len(z1[0]):] = z1.copy()
    Gamma = np.ones((len(U[0]), 1))
    W = inv(U @ U.T) @ U @ Gamma
    return np.array([W[0,0], W[1,0]]), W[2, 0]




def ACP_classifier(sample1, sample2):
    cnt = 0
    beta = 0.9
    W = np.array([[1, 1, 1]]).T
    Warr = [W.copy()]
    x1 = list(sample1[0,:])
    x2 = list(sample2[0,:])
    y1 = list(sample1[1,:])
    y2 = list(sample2[1,:])

    z = np.array([
        x1 + x2,
        y1 + y2,
        [1 for i in range(len(x1) * 2)],
        [-1 for i in range(len(x1))] + [1 for i in range(len(x1))]
    ])

    k = 0
    while True:
        ind = shuffle(list(range(400)))
        for i in range(400):
            j = ind[i]
            alpha = 1 / (k + 1) ** beta
            k += 1
            x = z[:3, j]
            x = np.array([[x[0], x[1], x[2]]]).T
            r = z[3, j]
            W = W + alpha * x @ sgn(r - W.T @ x)
        Warr.append(W)
        tmp0 = Warr[-2].flatten()
        tmp1 = Warr[-1].flatten()
        if norm(tmp1 - tmp0) < 0.001:
            break
        cnt += 1
    print(f"{cnt} поколений")
    return Warr


def get_y(x, W, wn):
    return -W[0] / W[1] * x - wn / W[1]


def experimental_probability(x0, x1, W, wn):
    s0 = 0
    s1 = 0

    N = x0.shape[1]
    for i in range(N):
        d0 = W.T @ x0[:,i] + wn
        d1 = W.T @ x1[:,i] + wn
        if d0 > 0:
            s0 += 1
        if d1 < 0:
            s1 += 1
    p0 = s0 / N
    p1 = s1 / N
    return p0, p1

####################
####################

M1 = np.array([0, 1])
M2 = np.array([1, -1])
M3 = np.array([-1, -1])

B1 = np.array([[0.2, 0.1],
               [0.1, 0.2]])

B2 = np.array([[0.3, 0],
               [0, 0.3]])

B3 = np.array([[0.7, -0.2],
               [-0.2, 0.9]])


sample1 = np.load("sample1_1.npy", allow_pickle=True)
sample2 = np.load("sample1_2.npy", allow_pickle=True)
bayes1 = np.load("bayes1.npy", allow_pickle=True)

print('B1=B2')
fisher_x = np.array([-1.5, 2.5])
W, wn = fisher_classifier1(B1, M1.T, M2.T)
fisher_y = get_y(fisher_x.T, W.T, wn)
p0, p1 = experimental_probability(sample1.T, sample2.T, W.T, wn)
print(f"Фишер: Экспериментальные ошибки: p0 = {p0}, p1 = {p1}")

# мин СКО
minstd_x = np.array([-1.5, 2.5])
W, wn = minstd_classifier(sample1.T, sample2.T)
minstd_y = get_y(minstd_x.T, W.T, wn)
p0, p1 = experimental_probability(sample1.T, sample2.T, W.T, wn)
print(f"Мин СКО: Экспериментальные ошибки: p0 = {p0}, p1 = {p1}")

# процедура Роббинса-Монро
acp_x1 = np.array([-1.5, 2.5])
Warr = ACP_classifier(sample1.T, sample2.T)
acp_y1 = get_y(acp_x1, [Warr[-1][0,0], Warr[-1][1,0]], Warr[-1][2,0])
p0, p1 = experimental_probability(sample1.T, sample2.T, np.array([Warr[-1][0,0], Warr[-1][1,0]]).T, Warr[-1][2,0])
print(f"ACP: Экспериментальные ошибки: p0 = {p0}, p1 = {p1}")



plt.scatter(sample1[:,0], sample1[:,1], color='blue', marker='.')
plt.scatter(sample2[:,0], sample2[:,1], color='red', marker='.')
plt.plot(bayes1[0,:], bayes1[1,:], color='green', label="Байес")
plt.plot(fisher_x, fisher_y, color='purple', label="Фишер")
plt.plot(minstd_x, minstd_y, color='brown', label="мин СКО")
plt.plot(acp_x1, acp_y1, color='orange', label="acp")
plt.grid(True)
plt.legend()
plt.show()
plt.close()



sample1 = np.load("sample2_1.npy", allow_pickle=True)
sample3 = np.load("sample2_3.npy", allow_pickle=True)
bayes2 = np.load("bayes2.npy", allow_pickle=True)

print('\nB1!=B2')
fisher_x = np.array([-2, 2])
W, wn = fisher_classifier2(B1, B3, M1.T, M3.T)
fisher_y = get_y(fisher_x.T, W.T, wn)
p0, p1 = experimental_probability(sample1.T, sample3.T, W.T, wn)
print(f"Фишер: Экспериментальные ошибки: p0 = {p0}, p1 = {p1}")

# мин СКО
minstd_x = np.array([-2, 2])
W, wn = minstd_classifier(sample1.T, sample3.T)
minstd_y = get_y(minstd_x.T, W.T, wn)
p0, p1 = experimental_probability(sample1.T, sample3.T, W.T, wn)
print(f"Мин СКО: Экспериментальные ошибки: p0 = {p0}, p1 = {p1}")

# процедура Роббинса-Монро
acp_x1 = np.array([-2, 2])
Warr = ACP_classifier(sample1.T, sample3.T)
acp_y1 = get_y(acp_x1, [Warr[-1][0,0], Warr[-1][1,0]], Warr[-1][2,0])
p0, p1 = experimental_probability(sample1.T, sample3.T, np.array([Warr[-1][0,0], Warr[-1][1,0]]).T, Warr[-1][2,0])
print(f"ACP: Экспериментальные ошибки: p0 = {p0}, p1 = {p1}")


plt.scatter(sample1[:,0], sample1[:,1], color='blue', marker='.')
plt.scatter(sample3[:,0], sample3[:,1], color='red', marker='.')
plt.scatter(bayes2[0,:], bayes2[1,:], color='green', marker=".", label="Байес")
plt.plot(fisher_x, fisher_y, color='purple', label="Фишер")
plt.plot(minstd_x, minstd_y, color='brown', label="мин СКО")
plt.plot(acp_x1, acp_y1, color='orange', label="acp")
plt.grid(True)
plt.legend()
plt.show()
plt.close()

