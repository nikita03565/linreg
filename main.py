import numpy as np
import matplotlib.pyplot as plt
from helpers import generate_data, LinearRegression

# data range and number of points
l_bound = 0
r_bound = 100
n = 1000
data = generate_data(l_bound, r_bound, n)

linreg = LinearRegression()
linreg.fit(data)

# Find regression line
xx = np.linspace(l_bound, r_bound, n)
yy = np.array(linreg.b[0] + linreg.b[1] * xx)

# Check predictions
pred_x = [[10], [33], [100], [50], [60], [70]]
pred = linreg.predict(pred_x)

plt.figure(1)
plt.plot(xx, yy.T, color='b')
plt.scatter(data[:, 0], data[:, 1], color='r')
plt.scatter(pred_x, pred, color='g')
plt.show()
