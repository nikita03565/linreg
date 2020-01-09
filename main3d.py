import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from helpers import generate_data3d, LinearRegression, calculate_rmse, SolveRidgeRegression, makeDFPlots, plotRMSEValue, getRMSEValues

# data range and number of points
l_bound = 0
r_bound = 100
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
print('rmse:', rmse)

wRR_list, df_list = SolveRidgeRegression(data[:, :-1], data[:, -1])
wRRArray = np.asarray(wRR_list)
dfArray = np.asarray(df_list)
makeDFPlots(dfArray, wRRArray)
plt.figure(1)
getRMSEValues(pred_x, actual_y, wRRArray, max_lamda=50, poly=1)
plt.show()
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], label='data')
ax.scatter(pred_x[:, 0], pred_x[:, 1], pred_y, color='m', label='pred')
ax.scatter(pred_x[:, 0], pred_x[:, 1], actual_y, color='c', label='actual')
ax.legend()

plt.show()
