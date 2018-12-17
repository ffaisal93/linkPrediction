import numpy as np

def DocumentCentrality(TD, n_it):
    n_it = 20
    TD = np.array(TD)
    N, T = np.shape(TD)
    if N > 1 and T > 1:
        l = np.zeros((N, n_it + 1))
        k = np.zeros((T, n_it + 1))
        k0 = TD.sum(axis=0)
        l0 = TD.sum(axis=1)
        k[:, 0] = k0
        l[:, 0] = l0

        for i in range(1, n_it + 1):
            k[:, i] = (1 / k0).transpose() * TD.transpose().dot(l[:, i - 1])
            l[:, i] = (1 / l0) * TD.dot(k[:, i - 1])
        x1 = k[:, 0]
        x1[x1 == 0] = np.nan
        x1 = np.log(x1)
        x1 = (x1 - np.nanmean(x1)) / np.nanstd(x1,ddof=1)
        x2 = k[:, 20]
        x2[x2 == 0] = np.nan
        x2 = (x2 - np.nanmean(x2)) / np.nanstd(x2,ddof=1)
        Cd = x1 + x2
        x1 = l[:, 0]
        x1[x1 == 0] = np.nan
        x1 = np.log(x1)
        x1 = (x1 - np.nanmean(x1)) / np.nanstd(x1,ddof=1)
        x2 = l[:, 20]
        x2[x2 == 0] = np.nan
        x2 = (x2 - np.nanmean(x2)) / np.nanstd(x2,ddof=1)
        Ct = x1 + x2
    else:
        print('Size of the input matrix is too small to do something useful.')
        Ct = []
        Cd = []
    return Cd, Ct


# if __name__ == '__main__':
#     TD = [[1, 2, 3, 4], [2, 3, 4, 5], [1, 2, 7, 8], [3, 5, 1, 2]]
#     TD = [[1, 2, 3], [4, 5, 6], [3, 5, 7], [7, 1, 2], [3, 6, 4], [3, 5, 5]]
#     Cd, Ct = DocumentCentrality(TD, 20)
#     print(Cd,Ct)
