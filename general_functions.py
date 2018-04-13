""" General functions used in this project"""

import numpy as np

def cosine_distance(x, y):
    return 1 - np.dot(x, y) / ((np.dot(x, x) * np.dot(y, y)) ** 0.5)

