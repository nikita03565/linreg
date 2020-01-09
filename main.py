import numpy as np
import matplotlib.pyplot as plt
from helpers import generate_data, LinearRegression, calculate_rmse, SolveRidgeRegression

# data range and number of points
l_bound = 0
r_bound = 10
n = 10
data = generate_data(l_bound, r_bound, n)

linreg = LinearRegression()
linreg.fit(data)

# Find regression line
xx = np.linspace(l_bound, r_bound, n)
yy = np.array(linreg.b[0] + linreg.b[1] * xx)

# Check predictions
check_data = generate_data(l_bound, r_bound, n // 10)
pred_x = [[x] for x in check_data[:, 0]]
actual_y = check_data[:, 1]
pred_y = linreg.predict(pred_x)

rmse = calculate_rmse(actual_y, pred_y)
print('rmse:', rmse)

# a, b = SolveRidgeRegression(data[:, 0], data[:, 1])
#print(a, b)

plt.figure(1)
plt.plot(xx, yy.T, color='tab:blue')
plt.scatter(data[:, 0], data[:, 1], color='c')
plt.scatter(check_data[:, 0], check_data[:, 1], color='r')
plt.scatter(pred_x, pred_y, color='m')
plt.show()
