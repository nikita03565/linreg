import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from helpers import generate_data3d, LinearRegression, calculate_rmse, make_coef_plot, RidgeRegression

# data range and number of points
l_bound = 0
r_bound = 1000
n = 1000
data = generate_data3d(l_bound, r_bound, n)

linreg = LinearRegression()
linreg.fit(data)


# Check predictions
check_data = generate_data3d(l_bound, r_bound, n)
pred_x = check_data[:, :-1]
actual_y = check_data[:, -1]
pred_y = linreg.predict(pred_x)

rmse = calculate_rmse(actual_y, pred_y)
print('rmse lin:', rmse)

rigreg = RidgeRegression()
rigreg.fit(data, 0.01)

rigpred_y = rigreg.predict(pred_x)

rmse_reg = calculate_rmse(actual_y, rigpred_y)
print('rmse reg:', rmse_reg)

make_coef_plot(data, check_data)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], label='data')
ax.scatter(pred_x[:, 0], pred_x[:, 1], pred_y, color='m', label='pred')
ax.scatter(pred_x[:, 0], pred_x[:, 1], rigpred_y, color='b', label='rigreg')
#ax.scatter(pred_x[:, 0], pred_x[:, 1], actual_y, color='c', label='actual')
ax.legend()

plt.show()
