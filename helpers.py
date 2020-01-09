import numpy as np
import matplotlib.pyplot as plt


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


class RidgeRegression:
    def __init__(self) -> None:
        self._b = None

    def fit(self, data: np.array, alpha=1.0) -> None:
        X = data[:, :-1]
        y = data[:, -1]
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        G = alpha * np.eye(X.shape[1])
        G[0, 0] = 0
        self._b = np.dot(np.linalg.inv(np.dot(X.T, X) + np.dot(G.T, G)), np.dot(X.T, y))

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return np.dot(X, self._b)

    @property
    def b(self):
        return self._b


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


def make_coef_plot(data, check_data):
    pred_x = check_data[:, :-1]
    actual_y = check_data[:, -1]

    alphas = np.linspace(0.00001, 1000, 10000)
    models = []
    rmses = []
    coefs = []
    for alpha in alphas:
        model = RidgeRegression()
        model.fit(data, alpha)
        pred_y = model.predict(pred_x)
        rmse = calculate_rmse(actual_y, pred_y)
        models.append(model)
        coefs.append(model.b)
        rmses.append(rmse)
    plt.figure(figsize=(15, 6))

    plt.subplot(121)
    ax = plt.gca()
    ax.plot(alphas, coefs)
    ax.set_xscale("log")
    plt.xlabel("alpha")
    plt.ylabel("b")
    plt.title("B")

    plt.subplot(122)
    ax = plt.gca()
    ax.plot(alphas, rmses)
    ax.set_xscale("log")
    plt.xlabel("alpha")
    plt.ylabel("error")
    plt.title("Error")
