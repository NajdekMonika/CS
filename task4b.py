from tools import *


min_X, max_X, min_Y, max_Y = 0.0, 3.0, 0.0, 3.0
x = np.arange(min_X, max_X, (max_X - min_X) / 50)
y = np.arange(min_Y, max_Y, (max_Y - min_Y) / 50)
starting_points = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)



results_X, results_Y = get_data_xy(starting_points, RK4_method, lotka_volterra_modified)




plt.figure(figsize=(20, 20))

for i in range(len(results_X)):
    plt.plot(results_X[i], results_Y[i])

plt.xlim(0.0, 10.0)
plt.ylim(0.0, 10.0)
plt.show()
plt.close()
