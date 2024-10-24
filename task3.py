from tools import *

a = [[-2, 1], [0, 2]]
b = [[3, -4], [2, -1]]
c = [[-3, -2], [-1, -3]]
d = [[2, 0], [0, 2]]
list_of_matrices = [a, b, c, d]
matrix_stats = []
for matrix in list_of_matrices:
    T = np.trace(matrix)
    D = np.linalg.det(matrix)
    val = T ** 2 / 4
    matrix_stats.append({'matrix': f'{matrix}', 'D': D, 'T': T, 'T**2/4': val, 'D<T**2/4?': D < val})

matrix_df = pd.DataFrame(matrix_stats)
print(matrix_df)

min_X, max_X, min_Y, max_Y = -3.4, 3.4, -3.4, 3.4
x = np.arange(min_X, max_X, (max_X - min_X) / 50)
y = np.arange(min_Y, max_Y, (max_Y - min_Y) / 50)
starting_points = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

X_list = []
Y_list = []

for matrix in list_of_matrices:
    my_fun = partial(diff_equ_matrix, m=matrix)
    results_X, results_Y = get_data_xy(starting_points, RK4_method, my_fun)
    X_list.append(results_X)
    Y_list.append(results_Y)

for idx in range(len(X_list)):
    x = X_list[idx]
    y = Y_list[idx]
    plt.figure(figsize=(20, 20))
    for i in range(len(x)):
        plt.plot(x[i], y[i])
    plt.xlim(-2.4, 2.4)
    plt.ylim(-2.4, 2.4)
    plt.show()
    plt.close()

