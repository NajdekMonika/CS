from tools import *

data = get_data([0.000001, 0.651, 1.999999], euler_method, 100)
print(data)


plt.figure(figsize=(10, 5))
for point in [0.0001, 0.91, 1.999]:
    time, fixed_points = euler_method(0, 10, 0.01, point, diff_equ)[:2]
    plt.plot(time, fixed_points, label=f"x$_0$: {point}")
plt.axhline(0, color='grey', lw=0.5, ls='--')
plt.axhline(1, color='grey', lw=0.5, ls='--')
plt.axhline(2, color='grey', lw=0.5, ls='--')
plt.legend()
plt.grid()
plt.show()

