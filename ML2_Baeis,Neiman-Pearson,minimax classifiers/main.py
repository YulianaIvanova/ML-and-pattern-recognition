import numpy as np
from numpy.linalg import inv, det
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import sqrt, log, erf
from scipy.special import erfinv


def mahalanobis(M0, M1, B):
    M_dif = M1 - M0
    res = M_dif.T @ inv(B) @ M_dif
    return res


def Ф(x):
    """
    Function Laplacian
    """
    if x < 0:
        res = -1 * erf(x / sqrt(2.0)) / 2.0
        #res = 1 - erf(abs(x)/ sqrt(2.0)) / 2.0
    else:
        res = erf(x / sqrt(2.0)) / 2.0
    return res


def wcf(M1, M2, B):
    """
    Wrong classification probability
    """
    p_m = mahalanobis(M1, M2, B)
    p0 = 1 - Ф(sqrt(p_m) / 2)
    p1 = Ф(- sqrt(p_m) / 2)

    return p0, p1


def bayes_classifier(x_arr, B, M1, M2, lambd=1):
    """
    Bayes classifier (same correlation matrix)
    """
    M_dif = M1 - M2
    M_sum = M1 + M2
    a = M_dif.T @ inv(B)
    b = -1 / 2 * M_sum.T @ inv(B) @ M_dif + np.log(lambd)

    #y = -13.33x0 + 16.66x1 + 6.66=0
    # x1 = 13.33/16.66*x0 - 6.66
    y_arr = -a[0] / a[1] * x_arr - b / a[1]

    return y_arr


def get_lambda(M1, M2, B, p0_fix=0.05):
    return np.exp(-0.5 * mahalanobis(M1, M2, B) + sqrt(mahalanobis(M1, M2, B)) * 1.645)  # 1.645


def experimental_probability(x_arr, M1, M2, B):
    s = 0

    N = x_arr.shape[1]
    for i in range(N):
        d1 = M1.T @ inv(B) @ x_arr[:,i] - 0.5 * M1.T @ inv(B) @ M1 + log(0.5)
        d2 = M2.T @ inv(B) @ x_arr[:,i] - 0.5 * M2.T @ inv(B) @ M2 + log(0.5)
        if d2 >= d1:
            s += 1
    p = s / N
    return p


def bayes_classifier_with_diff_disp(x_arr, B1, B2, M1, M2, lambd=1):
    """
    Bayes classifier (different correlation matrices)
    """
    B_div1 = inv(B2) - inv(B1)
    B_div2 = 2 * (M1.T @ inv(B1) - M2.T @ inv(B2))
    c = np.log(det(B2) / det(B1)) + 2 * np.log(lambd) - M1.T @ inv(B1) @ M1 + M2.T @ inv(B2) @ M2
    A = B_div1[1, 1]

    res = []
    for x in x_arr:
        B = B_div2[1] + x * (B_div1[1, 0] + B_div1[0, 1])
        C = c + B_div2[0] * x + x ** 2 * B_div1[0, 0]

        D = (B ** 2) - 4 * A * C
        if D >= 0:
            y1 = ((-1) * B + sqrt(D)) / (2 * A)
            y2 = ((-1) * B - sqrt(D)) / (2 * A)
            if y1 == y2:
                res.append([x, y1])
            else:
                res.append([x, y1])
                res.append([x, y2])
    return res

def correct_values(bound, min_val, max_val):
    idx = len(bound) - 1
    while True:
        if idx < 0:
            break
        if min_val > bound[idx][1] or bound[idx][1] > max_val:
            bound.pop(idx)
        idx -= 1

    return bound


def th_probability_with_diff_disp(x_arr, M1, M2, B1, B2):

    s = 0
    N = x_arr.shape[1]

    for i in range(N):
        dif1 = x_arr[:,i] - M1
        dif2 = x_arr[:,i] - M2
        d1 = log(1 / 3) - log(sqrt(det(B1))) - 1 / 2 * dif1.T @ inv(B1) @ dif1
        d2 = log(1 / 3) - log(sqrt(det(B2))) - 1 / 2 * dif2.T @ inv(B2) @ dif2
        if d2 >= d1:
            s += 1
    p = s / N
    return p, sqrt((1 - p) / (N * p))


def get_N(p, epsilon=0.05):
    return round((1 - p) / (epsilon ** 2 * p))

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

x0 = np.array([-1.5, 2.5])
y_bayes = bayes_classifier(x0,B1, M1.T, M2.T)
np.save("bayes1", np.array([x0, y_bayes], dtype=object))
y_pirson = bayes_classifier(x0, B1, M1.T, M2.T, get_lambda(M1.T, M2.T, B1))


print("Два класса с равными корреляционными матрицами")
# ошибочные вероятности: теоретические и экспериментальные
p0,p1 = wcf(M1.T, M2.T, B1)
print(f"Теоретические: p0={p0}, p1={p1} sum={p0 + p1}")

p0 = experimental_probability(sample1.T, M1.T, M2.T, B1)
p1 = experimental_probability(sample2.T, M2.T, M1.T, B1)
print(f"Экспериментальные: p0 = {p0}, p1 = {p1}")


plt.scatter(sample1[:,0], sample1[:,1], color='blue', marker='.')
plt.scatter(sample2[:,0], sample2[:,1], color='red', marker='.')
plt.plot(x0, y_bayes, color='green', label="Байес")
plt.plot(x0, y_pirson, color='cyan', label="Пирсон")
plt.grid(True)
plt.legend()
plt.show()
plt.close()

#########################

sample1 = np.load("sample2_1.npy", allow_pickle=True)
sample2 = np.load("sample2_2.npy", allow_pickle=True)
sample3 = np.load("sample2_3.npy", allow_pickle=True)

x0 = np.linspace(0.15, 3, 500) #заменить на от 0.15 до 3 (для красоты графиков)
bound_1_2 = bayes_classifier_with_diff_disp(x0, B1, B2, M1.T, M2.T)
bound_1_2 = correct_values(bound_1_2, -0.5, 3.5)# также заменить на -0.5 до 3.5
#b12_x = [i[0] for i in bound_1_2]
#b12_y = [i[1] for i in bound_1_2]
#np.save("bayes2", np.array([b12_x, b12_y], dtype=object))

x0 = np.linspace(-1.095, 2, 500) #0.2
bound_1_3 = bayes_classifier_with_diff_disp(x0, B1, B3, M1.T, M3.T)
b13_x = [i[0] for i in bound_1_3]
b13_y = [i[1] for i in bound_1_3]
np.save("bayes2", np.array([b13_x, b13_y], dtype=object))


x0 = np.linspace(-0.1, 1, 500)
bound_2_3 = bayes_classifier_with_diff_disp(x0, B2, B3, M2.T, M3.T)

print("\nТри класса с разными корреляционными матрицами")
print("Экспериментальные вероятности ошибочной классификации для 2 и 3 класса")
p0, epsilon0 = th_probability_with_diff_disp(sample2.T, M2.T, M3.T, B2, B3)
p1, epsilon1 = th_probability_with_diff_disp(sample3.T, M3.T, M2.T, B3, B2)
print(f"Для выборки 2 :p~ = {p0} epsilon = {epsilon0}")
print(f"Для выборки 3 :p~ = {p1} epsilon = {epsilon1}")
print(f"Для 2й выборки N >= {get_N(p0)}")
print(f"Для 3й выборки N >= {get_N(p1)}")

print("Экспериментальные вероятности ошибочной классификации для 1 и 3 класса")
p0, epsilon0 = th_probability_with_diff_disp(sample1.T, M1.T, M3.T, B1, B3)
p1, epsilon1 = th_probability_with_diff_disp(sample3.T, M3.T, M1.T, B3, B1)
print(f"Для выборки 1 :p~ = {p0}")
print(f"Для выборки 3 :p~ = {p1}")

plt.figure(figsize=(7, 7))
plt.scatter(sample1[:,0], sample1[:,1], color='blue', marker='.', label="1")
plt.scatter(sample2[:,0], sample2[:,1], color='red', marker='.', label="2")
plt.scatter(sample3[:,0], sample3[:,1], color='green', marker='.', label="3")
plt.scatter([i[0] for i in bound_1_2], [i[1] for i in bound_1_2], color="cyan", marker=".", label="1-2")
plt.scatter([i[0] for i in bound_1_3], [i[1] for i in bound_1_3], color="black", marker=".", label="1-3")
plt.scatter([i[0] for i in bound_2_3], [i[1] for i in bound_2_3], color="yellow", marker=".", label="2-3")
plt.grid(True)
# plt.xlim(-4, 4)
# plt.ylim(-4, 4)
plt.legend()
plt.show()
plt.close()