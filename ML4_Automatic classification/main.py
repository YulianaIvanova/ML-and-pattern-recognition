import numpy as np
import matplotlib.pyplot as plt
np.random.seed(10)


def gen_norms(M, B):
    A = np.zeros((2, 2), float)
    A[0, 0] = np.sqrt(B[0, 0])
    A[1, 0] = B[0, 1] / np.sqrt(B[0, 0])
    A[1, 1] = np.sqrt(B[1, 1] - pow(B[0, 1], 2) / B[0, 0])

    E = np.random.normal(0,1,2)

    X = A @ E.T + M
    return X


def gen_sample(M, B):
    sample = np.zeros((50,2), float)
    for i in range(50):
        x1 = gen_norms(M, B)
        sample[i, 0:2] = x1
    return sample


def check_B(B):
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            if np.abs(B[i,j]) > np.sqrt(B[i,i]*B[j,j]):
                return False
    return True


def euclidean(a: np.ndarray, b: np.ndarray):
    return np.linalg.norm(a - b)


def argmax(z, arg2):
    index_max = 0
    distance_max = euclidean(z[:, 0], arg2)
    for i in range(1, len(z[0])):
        if euclidean(z[:, i], arg2) > distance_max:
            distance_max = euclidean(z[:, i], arg2)
            index_max = i
    return index_max, distance_max


def maxmin_distance(z):
    y_dmin = []
    y_dtypical = []
    x_count_clusters = []
    avarage = np.array([np.mean(z[0]), np.mean(z[1])])
    ind0, _ = argmax(z, avarage)
    M = []
    M.append(np.array([z[0, ind0], z[1, ind0]]))
    ind1, _ = argmax(z, M[0])
    M.append(np.array([z[0, ind1], z[1, ind1]]))

    L = 3
    while True:
        clusters = []
        for i in range(len(M)):
            clusters.append([[], []])
        l_arr = []
        for i in range(len(z[0])):
            tmp_dist_arr = []
            for j in range(len(M)):
                tmp_dist_arr.append(euclidean(M[j], z[:, i]))
            l = min(tmp_dist_arr)
            l_arr.append(l)
            index = tmp_dist_arr.index(l)
            xi = z[:, i]
            clusters[index][0].append(xi[0])
            clusters[index][1].append(xi[1])

        d_cand = max(l_arr)
        d_cand_ind = l_arr.index(d_cand)
        cand = z[:, d_cand_ind]
        arr_d_min = []
        for i in range(len(M)):
            arr_d_min.append(euclidean(M[i], cand))
        d_min = min(arr_d_min)
        y_dmin.append(d_min)

        plt.figure(figsize=(8, 8))
        colors = ["lime", "dodgerblue", "firebrick", "gold", "indigo"]
        if len(clusters) > 5:
            raise NotImplementedError
        for i in range(len(clusters)):
            plt.scatter(clusters[i][0], clusters[i][1], marker=".", c=colors[i], label=f"c[{i}], {len(clusters[i][0])}")
            plt.scatter(M[i][0], M[i][1], c="black", marker="o")
        plt.scatter(cand[0], cand[1], c="red", marker='x')
        plt.grid(True)
        plt.legend()
        plt.show()

        arr_d_typical = []
        for i in range(len(M)):
            for j in range(i + 1, len(M)):
                arr_d_typical.append(euclidean(M[i], M[j]))
        d_typical = sum(arr_d_typical) / (len(arr_d_typical) * 2)
        y_dtypical.append(d_typical)
        # d_typical = sum(arr_d_typical) / ((L-1)*(L-2))

        plt.plot(arr_d_min, 'o')
        plt.plot([0, L - 2], [d_typical, d_typical])
        plt.show()
        plt.close()
        x_count_clusters.append(L - 1)
        if d_min > d_typical:
            M.append(np.array([cand[0], cand[1]]))
        else:
            break

        L += 1

    plt.figure(figsize=(8, 8))
    plt.plot(x_count_clusters, y_dmin, label="d_min")
    plt.plot(x_count_clusters, y_dtypical, label="d_typical")
    plt.legend()
    plt.show()
    print(len(M), "кластеров")


def intra_group_averages(z, K):
    amount = []
    r_arr = []
    R = [0 for _ in range(len(z[0]))]
    R_new = [0 for _ in range(len(z[0]))]
    M = []
    r = 1
    for i in range(K):
        i = int(np.random.uniform(0, len(z[0]) - 1))
        M.append(np.array([z[0, i], z[1, i]]))
    r += 1

    exit = False
    while not exit:
        r_arr.append(r)
        clusters = []
        for i in range(len(M)):
            clusters.append([[], []])
        l_arr = []
        for i in range(len(z[0])):
            tmp_dist_arr = []
            for j in range(len(M)):
                tmp_dist_arr.append(euclidean(M[j], z[:, i]))
            l = min(tmp_dist_arr)
            l_arr.append(l)
            index = tmp_dist_arr.index(l)
            xi = z[:, i]
            clusters[index][0].append(xi[0])
            clusters[index][1].append(xi[1])
            R_new[i] = index

        counter = 0
        for i in range(len(R_new)):
            if R[i] != R_new[i]:
                counter += 1
        amount.append(counter)

        plt.figure(figsize=(8, 8))
        colors = ["lime", "dodgerblue", "firebrick", "gold", "indigo"]
        for i in range(len(clusters)):
            plt.scatter(clusters[i][0], clusters[i][1], marker=".", c=colors[i], label=f"c[{i}], {len(clusters[i][0])}")
            plt.scatter(M[i][0], M[i][1], c="black", marker="o")

        plt.grid(True)
        plt.legend()
        plt.show()

        M_new = []

        for i in range(len(M)):
            sum = np.zeros(2)
            for j in range(len(clusters[i][0])):
                sum[0] += clusters[i][0][j]
                sum[1] += clusters[i][1][j]

            M_new.append(sum / len(clusters[i][0]))

        exit = True

        print(f"M[r-1]: {M}")
        print(f"M[r]: {M_new}\n\n")
        for i in range(len(M)):
            if not np.all(np.equal(M_new[i], M[i])):
                exit = False
                break
        if exit == False:
            M.clear()
            for i in range(len(M_new)):
                M.append(M_new[i].copy())
            M_new.clear()

        r += 1
    plt.figure(figsize=(10, 10))
    plt.title("Зависимость числа векторов признаков, сменивших номер кластера, от номера итерации")
    plt.plot(r_arr, amount)
    plt.show()


###############################
##############################

M1 = np.array([[0, 1]])
M2 = np.array([[1, -1]])
M3 = np.array([[-1, -1]])
M4 = np.array([[0, -2]])
M5 = np.array([[0, -3]])


B1 = np.array(
    [[0.08, 0.03],
     [0.03, 0.08]])
# dodgerblue
B2 = np.array(
    [[0.08, 0.001],
     [0.001, 0.08]])
# firebrick
B3 = np.array(
    [[0.08, 0.03],
     [0.03, 0.08]])
# gold
B4 = np.array(
    [[0.04, 0.0009],
     [0.0009, 0.04]])
# indigo
B5 = np.array(
    [[0.02, 0.001],
     [0.001, 0.02]])


print(check_B(B1))
print(check_B(B2))
print(check_B(B3))
print(check_B(B4))
print(check_B(B5))

# sample1 = gen_sample(M1,B1)
# sample2 = gen_sample(M2,B2)
# sample3 = gen_sample(M3,B3)
# sample4 = gen_sample(M4,B4)
# sample5 = gen_sample(M5,B5)

# np.save("sample1.npy", sample1, allow_pickle=True)
# np.save("sample2.npy", sample2, allow_pickle=True)
# np.save("sample3.npy", sample3, allow_pickle=True)
# np.save("sample4.npy", sample4, allow_pickle=True)
# np.save("sample5.npy", sample5, allow_pickle=True)

sample1 = np.load("sample1.npy", allow_pickle=True)
sample2 = np.load("sample2.npy", allow_pickle=True)
sample3 = np.load("sample3.npy", allow_pickle=True)
sample4 = np.load("sample4.npy", allow_pickle=True)
sample5 = np.load("sample5.npy", allow_pickle=True)


plt.scatter(sample1[:,0], sample1[:,1], c="lime", marker=".")
plt.scatter(sample2[:,0], sample2[:,1], marker=".", c= "dodgerblue")
plt.scatter(sample3[:,0], sample3[:,1], marker=".", c="firebrick")
plt.scatter(sample4[:,0], sample4[:,1], marker=".", c="gold")
plt.scatter(sample5[:,0], sample5[:,1], marker=".", c="indigo")
plt.grid(True)
plt.show()

z = np.vstack((sample1,sample2,sample3,sample4,sample5))
#maxmin_distance(z.T)
intra_group_averages(z.T, 5)