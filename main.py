import numpy as np
import matplotlib.pyplot as plt
from helpers import generate_data, LinearRegression

# data range and number of points
l_bound = 0
r_bound = 100
n = 100
data = generate_data(l_bound, r_bound, n)

linreg = LinearRegression()
linreg.fit(data)

# Find regression line
xx = np.linspace(l_bound, r_bound, n)
yy = np.array(linreg.b[0] + linreg.b[1] * xx)

# Check predictions
check_data = generate_data(l_bound, r_bound, n // 10)
pred_x = [[x] for x in check_data[:, 0]]
pred = linreg.predict(pred_x)

plt.figure(1)
plt.plot(xx, yy.T, color='b')
plt.scatter(data[:, 0], data[:, 1], color='r')
plt.scatter(check_data[:, 0], check_data[:, 1], color='g')
plt.scatter(pred_x, pred, color='c')
plt.show()
