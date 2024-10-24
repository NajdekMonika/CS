import warnings

import matplotlib.pyplot as plt
import numpy as np
from math import exp, log10
import pandas as pd
from math import sin
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

def diff_equ(arg):
    """
    Function to calculate the value of the differential equation
    Args:
       arg: argument

    Returns: value of the differential equation

    """
    return arg * (arg - 1) * (arg - 2)


def euler_method(t_p: float, t_k: float, h: float, x_0: float, fun: callable):
    """
     Function to calculate the fixed points using the Euler method

    Args:
       t_p: start time
       t_k: end time
       h: step
       x_0: starting point
       fun: differential equation

    Returns: time points, fixed points

    """
    t_euler = np.arange(t_p, t_k, h)
    x_euler = [x_0] * len(t_euler)
    for idx in range(1, len(t_euler)):
        x_euler[idx] = x_euler[idx - 1] + h * fun(x_euler[idx - 1])
    return t_euler, x_euler, x_euler[-1]


def get_data(points: list, method: callable, num_iterations=50):
    """
    Function to calculate the fixed points for different starting points
    Args:
       num_iterations:
       points: list of different starting points
       method: type of numerical method to calculate the fixed points

    Returns: DataFrame with fixed points for different starting points

    """
    results_4 = {point: [] for point in points}
    results_10 = {point: [] for point in points}
    for _ in range(num_iterations):
        for point in points:
            fixed_point_4 = method(0, 4, 0.01, point, diff_equ)[2]
            fixed_point_10 = method(0, 10, 0.01, point, diff_equ)[2]
            results_4[point].append(fixed_point_4)
            results_10[point].append(fixed_point_10)

    averages_4 = {point: np.mean(results_4[point]) for point in points}
    averages_10 = {point: np.mean(results_10[point]) for point in points}
    data = {"point": points, "average_result_4": [averages_4[point] for point in points], "average_result_10": [averages_10[point] for point in points]}
    df = pd.DataFrame(data)
    return df


def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)

def diff_equ2a(x, y):
    dxdt = y
    dydt = -x
    return [dxdt, dydt]


def diff_equ2b(x, y):
    dxdt = y
    dydt = -sin(x)
    return [dxdt, dydt]


def diff_equ2c(x, y):
    dxdt = y
    dydt = -x * (1 - x) * (1 + x)
    return [dxdt, dydt]


def diff_equ2d(x, y):
    dxdt = y
    dydt = x - x ** 3
    return [dxdt, dydt]


def RK4_method(t_p: int, t_k: int, h: float, x_0: float, y_0: float, fun: callable):
    """
    Function to calculate the fixed points using the Runge-Kutta method
    Args:
      y_0:
      t_p: start time
      t_k: end time
      h: step size
      x_0: starting point
      fun: function to calculate the value of the differential equation

    Returns: fixed points, time points, last fixed point

    """
    t_RK4 = np.arange(t_p, t_k, h)
    x_RK4 = [x_0] * len(t_RK4)
    y_RK4 = [y_0] * len(t_RK4)
    for t in range(len(t_RK4)):
        x_prev = x_RK4[t - 1]
        y_prev = y_RK4[t - 1]
        kx, ky = fun(x_prev, y_prev)
        X, Y = fun(x_prev + h * kx / 2, y_prev + h * ky / 2)
        x_RK4[t], y_RK4[t] = x_prev + h * X, y_prev + h * Y
    return x_RK4, y_RK4


def get_data_xy(points: np.ndarray, function: callable, equation: callable):
    """
    Function to calculate the fixed points for different starting points
    Args:
       equation:
       points: list of different starting points
       function: type of numerical method to calculate the fixed points

    Returns: DataFrame with fixed points for different starting points

    """
    results_x = []
    results_y = []
    for idxs in range(len(points)):
        xs, ys = function(0, 4, 0.01, points[idxs, 0], points[idxs, 1], equation)
        results_x.append(xs)
        results_y.append(ys)
    return results_x, results_y


def diff_equ_matrix(x, y, m):
    dxdt = x * m[0][0] + y * m[0][1]
    dydt = x * m[1][0] + y * m[1][1]
    return [dxdt, dydt]


def determine_type(det, tr):
    pass


def diff_equ_matA(x, y):
    dxdt = -2 * x + 1 * y
    dydt = 2 * y
    return [dxdt, dydt]


def diff_equ_matB(x, y):
    dxdt = 3 * x - 4 * y
    dydt = 2 * x - 1 * y
    return [dxdt, dydt]


def diff_equ_matC(x, y):
    dxdt = -3 * x - 2 * y
    dydt = -1 * x - 3 * y
    return [dxdt, dydt]


def diff_equ_matD(x, y):
    dxdt = 2 * x
    dydt = 2 * y
    return [dxdt, dydt]


def lotka_volterra(x, y):
    dxdt = x * (3 - x - 2 * y)
    dydt = y * (2 - x - y)
    return [dxdt, dydt]


def lotka_volterra_modified(x, y):
    # a, b, c, d = 1.5, 1, 1, 3
    a, b, c, d = 1, 1, 3, 1
    dxdt = x * (a - b * y)
    dydt = y * (-c + d * x)
    return [dxdt, dydt]
