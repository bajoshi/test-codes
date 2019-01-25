from __future__ import division

import numpy as np

def clip(a, min_value, max_value):
    min_les_idx = np.where(a < min_value)
    a[min_les_idx] = min_value
    max_gtr_idx = np.where(a > max_value)
    a[max_gtr_idx] = max_value
    return a

def compute(array_1, array_2, a, b, c):
    """
    This function must implement the formula
    np.clip(array_1, 2, 10) * a + array_2 * b + c

    array_1 and array_2 are 2D.
    """
    x_max = array_1.shape[0]
    y_max = array_1.shape[1]

    assert array_1.shape == array_2.shape

    result = np.zeros((x_max, y_max), dtype=array_1.dtype)

    for x in range(x_max):
        for y in range(y_max):
            tmp = clip(array_1, 2, 10)
            tmp = tmp * a + array_2[x, y] * b
            result[x, y] = tmp + c

    return result