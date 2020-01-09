import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from helpers import generate_data3d, LinearRegression3d, calculate_rmse, SolveRidgeRegression

# data range and number of points
l_bound = 0
r_bound = 10
n = 10
data = generate_data3d(l_bound, r_bound, n)

linreg = LinearRegression3d()
linreg.fit(data)


# Check predictions
check_data = generate_data3d(l_bound, r_bound, n)
pred_x = check_data[:, :-1]
actual_y = check_data[:, -1]
pred_y = linreg.predict(pred_x)

xx1 = np.linspace(l_bound, r_bound, n)
xx2 = np.linspace(l_bound, r_bound, n)
yy = np.array(linreg.b[0] + linreg.b[1] * xx1 + linreg.b[2] * xx2)

rmse = calculate_rmse(actual_y, pred_y)
print('rmse:', rmse)


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], label='data')
ax.plot(xx1, xx2, yy, color='tab:blue')
ax.scatter(pred_x[:, 0], pred_x[:, 1], pred_y, color='m', label='pred')
ax.scatter(pred_x[:, 0], pred_x[:, 1], actual_y, color='c', label='actual')
ax.legend()

plt.show()

