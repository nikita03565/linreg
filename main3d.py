import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from helpers import generate_data3d, LinearRegression3d, calculate_rmse, SolveRidgeRegression

# data range and number of points
l_bound = 0
r_bound = 100
n = 1000
data = generate_data3d(l_bound, r_bound, n)
xs = data[:, :-1]
y = data[:, -1]
#print('xs', xs.shape, xs)
#print('data', data.shape, data)
m = np.shape(xs)[0]
ones = np.ones(m).reshape((m, 1))
#print('ones', ones.shape, ones)
res = np.append(ones, xs, axis=1)
#print('res', res)

linreg = LinearRegression3d()

linreg.fit(data)


xx1 = np.linspace(l_bound, r_bound, n)
xx2 = np.linspace(l_bound, r_bound, n)
yy = np.array(linreg.b[0] + linreg.b[1] * xx1 + linreg.b[2] * xx2)


print(linreg.b)





fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], label='data')

ax.plot(xx1, xx2, yy, color='tab:blue')
ax.legend()

plt.show()
# linreg = LinearRegression()
# linreg.fit(data)
#
# # Find regression line
# xx = np.linspace(l_bound, r_bound, n)
# yy = np.array(linreg.b[0] + linreg.b[1] * xx)
#
# # Check predictions
# check_data = generate_data(l_bound, r_bound, n // 10)
# pred_x = [[x] for x in check_data[:, 0]]
# actual_y = check_data[:, 1]
# pred_y = linreg.predict(pred_x)
#
# rmse = calculate_rmse(actual_y, pred_y)
# print('rmse:', rmse)
#
# a, b = SolveRidgeRegression(data[:, 0], data[:, 1])
# print(a, b)
#
# plt.figure(1)
# plt.plot(xx, yy.T, color='tab:blue')
# plt.scatter(data[:, 0], data[:, 1], color='c')
# plt.scatter(check_data[:, 0], check_data[:, 1], color='r')
# plt.scatter(pred_x, pred_y, color='m')
# plt.show()
