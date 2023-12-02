# Standard library imports:
import numpy as np
# Import chebpts2
from chebpts2 import chebpts2
"""
Created on Wed Dec  1 08:22:31 2023

Copyright 2023 by Gentian Zavalani.
"""
def transform(n,duffy_transform=False):
    """
    Transform Chebyshev points from [-1, 1]^2 to a reference simplex.

    Parameters:
        unisolvent_nodes (ndarray): Chebyshev points on the square.
        duffy_transform (bool): Whether to apply Duffy's transform.

    Returns:
        ndarray: Transformed points on the simplex.
    """
    [u,v]=chebpts2(n,n, kind=2)
    x = u.ravel()
    y = v.ravel()

    # Transformation to the reference simplex
    if duffy_transform:
        points_simplex_x = (1/4) * ((1 + x) * (1 - y))
        points_simplex_y = (1 + y) / 2
    else:
        points_simplex_x = (1 + x) * (3 - y) / 8
        points_simplex_y = (3 - x) * (y + 1) / 8

    return np.column_stack((points_simplex_x, points_simplex_y))
    return points_simplex