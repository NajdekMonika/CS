import matplotlib
import matplotlib.pyplot as plt
from tools import *

min_X, max_X, min_Y, max_Y = -3.5, 3.5, -3.5, 3.5
x = np.arange(min_X, max_X, (max_X - min_X) / 50)
y = np.arange(min_Y, max_Y, (max_Y - min_Y) / 50)
starting_points = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)


X_list = []
Y_list = []
for method in [diff_equ2a, diff_equ2b, diff_equ2c, diff_equ2d]:
    results_X, results_Y = get_data_xy(starting_points, RK4_method, method)
    X_list.append(results_X)
    Y_list.append(results_Y)

for idx in range(len(X_list)):
    x = X_list[idx]
    y = Y_list[idx]
    plt.figure(figsize=(20, 20))
    for i in range(len(x)):
        plt.plot(x[i], y[i])

    if idx == 0:
        plt.axvline(x=0, color='k', linestyle='--')
        plt.axhline(y=0, color='k', linestyle='--')
    elif idx == 1:
        plt.axvline(x=-3.2, color='k', linestyle='--')
        plt.axvline(x=0, color='k', linestyle='--')
        plt.axhline(y=0, color='k', linestyle='--')
        plt.axvline(x=3.2, color='k', linestyle='--')
    elif idx == 2:
        plt.axvline(x=0, color='k', linestyle='--')
        plt.axhline(y=0, color='k', linestyle='--')
        plt.axvline(x=-1, color='k', linestyle='--')
        plt.axvline(x=1, color='k', linestyle='--')
    elif idx == 3:
        plt.axhline(y=0, color='k', linestyle='--')
        plt.axvline(x=0, color='k', linestyle='--')
        plt.axvline(x=-1, color='k', linestyle='--')
        plt.axvline(x=1, color='k', linestyle='--')
    plt.xlim(-3.5, 3.5)
    plt.ylim(-3.5, 3.5)
    plt.show()
    plt.close()


