from tools import *
import matplotlib.cm as cm
from matplotlib.colors import Normalize


min_X, max_X, min_Y, max_Y = -0.2, 3.4, -0.2, 3.4
x = np.arange(min_X, max_X, (max_X - min_X) / 50)
y = np.arange(min_Y, max_Y, (max_Y - min_Y) / 50)
starting_points = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)



results_X, results_Y = get_data_xy(starting_points, RK4_method, lotka_volterra)




plt.figure(figsize=(20, 20))

for i in range(len(results_X)):
    plt.plot(results_X[i], results_Y[i])

plt.xlim(-0.2, 3.4)
plt.ylim(-0.2, 3.4)
plt.show()
plt.close()
