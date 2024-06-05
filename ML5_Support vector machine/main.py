import numpy as np
from matplotlib import pyplot as plt
from qpsolvers import solve_qp
from sklearn import svm
from sklearn.model_selection import train_test_split
from cvxopt import matrix, solvers
from sklearn.svm import SVC
from tqdm import tqdm

def get_y(x, W, wn):
    return [-W[0] / W[1] * i - wn / W[1] for i in x]


def get_dataset(s1, s2):
    new_shape = (s1.shape[0] + 1, s1.shape[1] * 2)
    new_dataset = np.zeros(new_shape)
    number = 0
    i = 0
    while i != new_dataset.shape[1]:
        new_dataset[0, i] = s1[0, number]
        new_dataset[1, i] = s1[1, number]
        new_dataset[2, i] = -1
        i += 1
        new_dataset[0, i] = s2[0, number]
        new_dataset[1, i] = s2[1, number]
        new_dataset[2, i] = 1
        i += 1
        number += 1
    return new_dataset


def get_P(z):
    n = z.shape[1]
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ri, rj = z[2, i], z[2, j]
            xi_x, xi_y = z[0, i], z[1, i]
            xj_x, xj_y = z[0, j], z[1, j]
            P[i, j] = (ri * rj) * (xi_x * xj_x + xi_y * xj_y)
    return P


def get_support_vectors(dataset, lambdas):
    """Нахождение опорных векторов"""
    support_vectors = []
    support_lambdas = []
    for i in range(0, len(lambdas)):
        if lambdas[i] > 0.0001:
            support_vectors.append(dataset[:, i])
            support_lambdas.append(lambdas[i])
    return np.array(support_vectors).T, np.array(support_lambdas)


def separate_sup_vectors(vectors):
    """Разделение опорных векторов по классам"""
    class_0_vectors = []
    class_1_vectors = []
    tmp = np.transpose(vectors)
    for vec in tmp:
        if vec[2] == -1.0:
            class_0_vectors.append(vec[0:2])
        elif vec[2] == 1.0:
            class_1_vectors.append(vec[0:2])
    return np.array(class_0_vectors).T, np.array(class_1_vectors).T


def get_classifier_parameters(s_v, lambdas):
    """Возвращает параметры линейного классификатора: W и wn"""
    W = np.zeros(2)
    x = np.array([s_v[0], s_v[1]])
    r = s_v[2]
    for i in range(len(lambdas)):
        W = W + x[:, i] * lambdas[i] * r[i]

    wn = 0
    for i in range(len(s_v[0])):
        wn += r[i] - W.reshape(1, 2) @ x[:, i].reshape(2, 1)
    wn = wn[0, 0]

    return W, wn / len(s_v[0])


def draw_results(title, sample1, sample2, W_array, labels, sup1=None, sup2=None):
    colors = ["maroon", "olive", "steelblue"]
    plt.figure(figsize=(8, 8))
    plt.title(title)
    plt.scatter(sample1[0], sample1[1], c="blue", marker="x", linewidths=1)
    plt.scatter(sample2[0], sample2[1], c="red", marker="+", linewidths=1)
    if sup1 is not None and sup2 is not None:
        plt.scatter(sup1[0], sup1[1], c="lime", marker='o', zorder=0.5)
        plt.scatter(sup2[0], sup2[1], c="yellow", marker='o', zorder=0.5)

    for i in range(len(W_array)):
        W = W_array[i, 0]
        wn = W_array[i, 1]
        #print(f'W = {W}, wn = {wn}')
        border_x = np.array([-2, 2])
        border_y = np.array(get_y(border_x, [W[0], W[1]], wn[0, 0] if isinstance(wn, np.ndarray) else wn))
        plt.plot(border_x, border_y, color=colors[i], linewidth=2, label=labels[i])
        plt.plot(border_x + 1 / W[0], border_y, color=colors[i], linewidth=1, linestyle="dashed")
        plt.plot(border_x - 1 / W[0], border_y, color=colors[i], linewidth=1, linestyle="dashed")
    plt.legend()
    plt.show()


def separate_sup_vectors_with_indexes(dataset, indexes):
    """Разделение определенных опорных векторов по классам"""
    class_0_vectors = []
    class_1_vectors = []
    for index in indexes:
        vect = dataset[:, index]
        if vect[2] == -1.0:
            class_0_vectors.append(vect[0:2])
        else:
            class_1_vectors.append(vect[0:2])
    return np.transpose(class_0_vectors), np.transpose(class_1_vectors)


def task2(sample1, sample2):
    z = get_dataset(sample1, sample2)
    P = get_P(z)
    A = z[2, :]
    q = -1 * np.ones(z.shape[1])
    b = np.zeros(1)
    G = np.eye(z.shape[1]) * -1
    h = np.zeros((z.shape[1],))
    lambdas = solve_qp(P, q, G, h, A, b, solver='cvxopt')

    support_vectors, support_lambdas = get_support_vectors(z, lambdas)
    sup_0_qp, sup_1_qp = separate_sup_vectors(support_vectors)
    # print(support_vectors)
    # print(sup_0_qp)
    w_qp, wn_qp = get_classifier_parameters(support_vectors, support_lambdas)
    #W_qp = np.asarray(t).transpose()
    W_qp = np.array([[w_qp, wn_qp]], dtype=object)
    #W_qp = np.concatenate((w_qp, [wn_qp]))
    draw_results("Квадратичное уравнение", sample1, sample2, W_qp, ['qp'], sup_0_qp, sup_1_qp)

    clf_svc = svm.SVC(kernel="linear", C=1)
    X = z[0:2, :].T
    r = z[2, :]
    clf_svc.fit(X, r)
    support_vectors_ind = clf_svc.support_
    w_svc = clf_svc.coef_.T
    wn_svc = clf_svc.intercept_[0]
    W_svc = np.array([[w_svc, wn_svc]], dtype=object)
    sup_0_svc, sup_1_svc = separate_sup_vectors_with_indexes(z, support_vectors_ind)
    draw_results("SVC", sample1, sample2, W_svc, ['svc'], sup_0_svc, sup_1_svc)

    clf_lin = svm.LinearSVC()
    clf_lin.fit(X, r)
    w_linear = clf_lin.coef_.T
    wn_linear = clf_lin.intercept_[0]
    W_lin = np.array([[w_linear, wn_linear]], dtype=object)
    draw_results("SVC linear", sample1, sample2, W_lin, ['lin_svc'])
    W_array = np.array([[w_qp, wn_qp], [w_svc, wn_svc], [w_linear, wn_linear]], dtype=object)
    draw_results("Сравнение", sample1, sample2, W_array, ['qp', 'svc', 'lin_svc'])


def get_errors(z, W, wn):
    count_errors0 = 0
    count_errors1 = 0
    W = W.T
    for i in range(0, z.shape[1], 2):
        xi = z[0:2, i]
        if np.sign(W @ xi + wn) != np.sign(z[2, i]):
            count_errors0 += 1
    for i in range(1, z.shape[1], 2):
        xi = z[0:2, i]
        if np.sign(W @ xi + wn) != np.sign(z[2, i]):
            count_errors1 += 1
    return count_errors0 / 100, count_errors1 / 100


def task_3(s1, s2):
    z = get_dataset(s1, s2)
    z_len = z.shape[1]
    P = get_P(z)
    A = z[2, :]
    q = -1 * np.ones(z.shape[1])
    b = np.zeros(1)
    G = np.concatenate((np.eye(z_len) * -1, np.eye(z_len)), axis=0)
    X = np.transpose(z[0:2, :])
    Y = z[2, :]
    qp_errors_array = []
    svc_errors_array = []
    for C in [0.1, 1, 4, 10]:
        h = np.concatenate((np.zeros((z_len,)), np.full((z_len,), C)), axis=0)

        lambdas = solve_qp(P, q, G, h, A, b, solver='cvxopt')
        support_vectors, support_lambdas = get_support_vectors(z, lambdas)
        sup_0_qp, sup_1_qp = separate_sup_vectors(support_vectors)
        W, wn = get_classifier_parameters(support_vectors, support_lambdas)
        W_arr = np.array([[W, wn]], dtype=object)
        qp_errors_array.append(get_errors(z, W, wn))
        draw_results(f'C = {C} qp hyperplane', s1, s2, W_arr, ['qp'], sup_0_qp, sup_1_qp)


        clf_svc = svm.SVC(kernel="linear", C=C)
        clf_svc.fit(X, Y)
        support_vectors_ind = clf_svc.support_
        W = clf_svc.coef_.T
        wn = clf_svc.intercept_[0]
        W_arr = np.array([[W, wn]], dtype=object)
        sup_0_svc, sup_1_svc = separate_sup_vectors_with_indexes(z, support_vectors_ind)
        draw_results(f'C = {C} svc hyperlane', s1, s2, W_arr, ['svc'], sup_0_svc, sup_1_svc)
        svc_errors_array.append(get_errors(z, W, wn))

    qp_errors_array = np.array(qp_errors_array)
    svc_errors_array = np.array(svc_errors_array)
    C = np.array([0.1, 1, 4, 10])
    for i in range(len(qp_errors_array)):
        print(f'С = {C[i]} qp error : {qp_errors_array[i]} svc error : {svc_errors_array[i]}')

def get_K(x, y, K, K_params):
    if K == 'poly':  # полиномиальное однородное
        c = K_params[0]
        d = K_params[1]
        if x.shape != (1, 2):
            x = x.reshape(1, 2)
        tmp = x @ y + c
        return pow(tmp, d)
    elif K == 'rbf':  # радиальная функция
        gamma = K_params[0]
        return np.exp(-gamma * np.sum(np.power((x - y), 2)))
    elif K == 'sigmoid':  # сигмодальная функция
        if x.shape != (1, 2):
            x = x.reshape(1, 2)
        gamma = K_params[0]
        c = K_params[1]
        return np.tanh(gamma * np.matmul(x, y)[0] + c)
    return None


def get_P_kernel(dataset, K, K_params):
    N = dataset.shape[1]
    P = np.ndarray(shape=(N, N))
    for i in range(0, N):
        for j in range(0, N):
            P[i, j] = dataset[2, j] * dataset[2, i] * get_K(dataset[0:2, j], dataset[0:2, i], K, K_params)
    return P


def get_discriminant_kernel(support_vectors, lambda_r, x, K, K_params):
    sum = 0
    for j in range(support_vectors.shape[1]):
        sum += lambda_r[j] * get_K(support_vectors[0:2, j].reshape(2, 1), x, K, K_params)
    return sum


def get_svc(C, K, K_params):
    if K == 'poly':
        return SVC(C=C, kernel=K, degree=K_params[1], coef0=K_params[0])
    if K == 'rbf':
        return SVC(C=C, kernel=K, gamma=K_params[0])  # radial
    if K == 'sigmoid':
        return SVC(C=C, kernel=K, coef0=K_params[1], gamma=K_params[0])


def task4(samples0, samples1, kernel_name, kernel_params):
    z = get_dataset(samples0, samples1)
    z_len = z.shape[1]
    P = get_P_kernel(z, kernel_name, kernel_params)
    A = z[2, :]
    q = np.full((z_len, 1), -1, dtype=np.double)
    b = np.zeros(1)
    G = np.concatenate((np.eye(z_len) * -1, np.eye(z_len)), axis=0)
    X = np.transpose(z[0:2, :])
    r = z[2, :]
    eps = 1e-04
    qp_errors_array = []
    svc_errors_array = []

    for C in [0.1, 1, 10]:
        h = np.concatenate((np.zeros((z_len,)), np.full((z_len,), C)), axis=0)
        lambdas = solve_qp(P, q, G, h, A, b, solver='cvxopt')
        # print("qp solved")
        support_vectors_ind = lambdas > eps
        support_vectors, _ = get_support_vectors(z, lambdas)
        sup_0_qp, sup_1_qp = separate_sup_vectors(support_vectors)
        r_arr = support_vectors[2, :]
        wn = []
        for j in range(support_vectors.shape[1]):
            wn.append(
                get_discriminant_kernel(
                    support_vectors,
                    (lambdas * A)[support_vectors_ind],
                    support_vectors[0:2, j].reshape(2, 1),
                    kernel_name,
                    kernel_params
                )
            )
        wn = np.mean(r_arr - np.array(wn))
        # print("wn calculated")
        p0 = 0.
        p1 = 0.

        for i in range(samples0.shape[1]):
            if get_discriminant_kernel(support_vectors, (lambdas * A)[support_vectors_ind], samples0[0:2, i],
                                       kernel_name, kernel_params) + wn > 0:
                p0 += 1
            if get_discriminant_kernel(support_vectors, (lambdas * A)[support_vectors_ind], samples1[0:2, i],
                                       kernel_name, kernel_params) + wn < 0:
                p1 += 1
        p0 /= samples0.shape[1]
        p1 /= samples1.shape[1]
        qp_errors_array.append(p0 + p1)
        # print("errors calculated")

        y = np.linspace(-4, 4, z_len)
        x = np.linspace(-4, 4, z_len)
        xx, yy = np.meshgrid(x, y)
        xy = np.vstack((xx.ravel(), yy.ravel())).T
        Z = []
        for i in tqdm(range(xy.shape[0]), f"disc func, C = {C}"):
            Z.append(
                get_discriminant_kernel(
                    support_vectors,
                    (lambdas * A)[support_vectors_ind],
                    xy[i].reshape(2, 1),
                    kernel_name, kernel_params)
                + wn)
        Z = np.array(Z).reshape(xx.shape)

        show_contours("quadratic", samples0, samples1, xx, yy, Z, C, kernel_name, sup_0_qp, sup_1_qp)

        X_train, X_test, r_train, r_test = train_test_split(X, r, test_size=0.5, random_state=42)
        clf = get_svc(C, kernel_name, kernel_params)

        clf.fit(X_train, r_train)
        support_vectors_ind = clf.support_
        sup_0_svc, sup_1_svc = separate_sup_vectors_with_indexes(z, support_vectors_ind)

        y = np.linspace(-4, 4, z_len)
        x = np.linspace(-4, 4, z_len)
        xx, yy = np.meshgrid(x, y)
        xy = np.vstack((xx.ravel(), yy.ravel())).T
        # discriminant_func_values_svc = clf.decision_function(xy).reshape(xx.shape)
        show_contours("svc", samples0, samples1, xx, yy, Z, C, kernel_name, sup_0_svc, sup_1_svc)

        errors_count = 0
        r_preds = clf.predict(X_test)
        for i in range(0, len(r_preds)):
            if r_preds[i] != r_test[i]:
                errors_count += 1
        # print(f"{errors_count}/{len(X_test)} = {errors_count / len(X_test)}")
        svc_errors_array.append(errors_count / len(X_test))

    qp_errors_array = np.array(qp_errors_array)
    svc_errors_array = np.array(svc_errors_array)
    C = np.array([0.1, 1, 10])
    for i in range(0, len(qp_errors_array)):
        print(f'С = {C[i]}, qp error = {qp_errors_array[i]}, svc error = {svc_errors_array[i]}')


def show_contours(method, samples1, samples2, xx, yy, Z, C, kernel, sup1=None, sup2=None):
    plt.title(f"{method}, C = {C}, K = ({kernel})")
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)

    plt.scatter(samples1[0], samples1[1], marker='+')
    plt.scatter(samples2[0], samples2[1], marker='x')
    if sup1 is not None and sup2 is not None:
        plt.scatter(sup1[0], sup1[1], c="lime", marker='o', zorder=0.5)
        plt.scatter(sup2[0], sup2[1], c="yellow", marker='o', zorder=0.5)

    plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['maroon', 'black', 'steelblue'])
    plt.show()
    plt.close()


if __name__ == "__main__":
    # task1
    sample1 = np.load("sample1_1.npy", allow_pickle=True)
    sample2 = np.load("sample1_2.npy", allow_pickle=True)
    #task2(sample1.T, sample2.T)

    sample3 = np.load("sample2_1.npy", allow_pickle=True)
    sample4 = np.load("sample2_2.npy", allow_pickle=True)
    task_3(sample3.T, sample4.T)

    kernel_array = np.array(['poly', 'rbf', 'sigmoid'])
    kernel_params_array = np.array([[1, 2], [0.3], [0.1, -1]], dtype=object)

    for i in range(0, len(kernel_array)):
        print(kernel_array[i], "kernel")
        #task4(sample3.T, sample4.T, kernel_array[i], kernel_params_array[i])


