import numpy as np
import matplotlib.pyplot as plt


def calculate_rmse(actual, predicted):
    return np.sqrt(np.mean((actual-predicted)**2))


def generate_data(l_bound: float, r_bound: float, n: int) -> np.array:
    x = np.linspace(l_bound, r_bound, n)
    delta = np.random.uniform(-5, 5, x.size)
    y = 0.4 * x + 3 + delta
    return np.array(list(zip(x, y)))


def generate_data3d(l_bound: float, r_bound: float, n: int) -> np.array:
    x1 = np.linspace(l_bound, r_bound, n)
    delta_x2 = np.random.uniform(-3, 3, x1.size)
    x2 = 0.7 * x1 + 3 + delta_x2
    delta_y = np.random.uniform(-5, 5, x1.size)
    y = 0.4 * x1 + 1.1 * x2 + 3 + delta_y
    return np.array(list(zip(x1, x2, y)))


class LinearRegression:
    def __init__(self) -> None:
        self._b = None

    def fit(self, data: np.array) -> None:
        m = np.shape(data)[0]
        x = np.append(np.ones(m).reshape((m, 1)), data[:, :-1], axis=1)
        y = np.array(data[:, -1]).T
        p_mat = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
        self._b = np.copy(p_mat)

    def predict(self, x_pred: list) -> list:
        if self._b is None:
            raise Exception('Not trained yet')
        y_res = []
        for x in x_pred:
            _x = [1, *x]
            if len(_x) != len(self._b):
                raise Exception('Mismatch')
            y = sum(new_x * b for new_x, b in zip(_x, self._b))
            y_res.append(y)
        return y_res

    @property
    def b(self):
        return self._b


def SolveRidgeRegression(X, y):
    wRR_list = []
    df_list = []
    for i in range(1000):
        print(i)
        lam_par = i
        xtranspose = np.transpose(X)
        xtransx = np.dot(xtranspose, X)
        if xtransx.shape[0] != xtransx.shape[1]:
            raise ValueError('Needs to be a square matrix for inverse')
        lamidentity = np.identity(xtransx.shape[0]) * lam_par
        matinv = np.linalg.inv(lamidentity + xtransx)
        xtransy = np.dot(xtranspose, y)
        wRR = np.dot(matinv, xtransy)
        _, S, _ = np.linalg.svd(X)
        df = np.sum(np.square(S) / (np.square(S) + lam_par))
        wRR_list.append(wRR)
        df_list.append(df)
    return wRR_list, df_list


def makeDFPlots(dfArray, wRRArray):
    #print wRR_array.shape, df_array.shape
    plt.figure()
    colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628']
    labels = ["Dimension 1", "Dimension 2", "Dimension 3", "Dimension 4 (car weight)", "Dimension 5", "Dimension 6 (car year)", "Dimension 7"]
    for i in range(0, wRRArray[0].shape[0]):
        plt.plot(dfArray, wRRArray[:,i], color = colors[i])
        plt.scatter(dfArray, wRRArray[:,i], color = colors[i], s = 8, label=labels[i])
    # df(lambda)
    plt.xlabel(r"df($\lambda$)")
    # and a legend
    plt.legend(loc='lower left')
    plt.show()


def plotRMSEValue(max_lamda, RMSE_list, poly):
    colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628']
    legend = ["Polynomial Order, p = 1", "Polynomial Order, p = 2", "Polynomial Order, p = 3"]
    plt.plot(range(len(RMSE_list)), RMSE_list, color = colors[poly])
    plt.scatter(range(len(RMSE_list)), RMSE_list, color = colors[poly] , s = 8, label= legend[poly-1])
    # df(lambda)
    plt.xlabel(r"$\lambda$")
    plt.ylabel("RMSE")
    # and a legend
    plt.legend(loc='upper left')
    plt.title(r"RMSE vs $\lambda$ values for the test set, $\lambda$ = 0..%d"%(max_lamda))


def getRMSEValues(X_test, y_test, wRRArray, max_lamda, poly):
    RMSE_list = []
    for lamda in range(0, max_lamda+1):
        wRRvals = wRRArray[lamda]
        y_pred = np.dot(X_test, wRRvals)
        RMSE = np.sqrt(np.sum(np.square(y_test - y_pred))/len(y_test))
        RMSE_list.append(RMSE)
    plotRMSEValue(max_lamda, RMSE_list, poly=poly)
