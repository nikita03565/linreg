import numpy as np


def calculate_rmse(actual, predicted):
    return np.sqrt(np.mean((actual-predicted)**2))


def generate_data(l_bound: float, r_bound: float, n: int) -> np.array:
    x = np.linspace(l_bound, r_bound, n)
    delta = np.random.uniform(-5, 5, x.size)
    y = 0.4 * x + 3 + delta
    return np.array(list(zip(x, y)))


class LinearRegression:
    def __init__(self) -> None:
        self._b = None

    def fit(self, data: np.array) -> None:
        m = np.shape(data)[0]
        x = np.array([np.ones(m), data[:, 0]]).T
        y = np.array(data[:, 1]).T
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
