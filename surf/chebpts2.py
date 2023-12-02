"""
Created on Wed Dec  1 08:22:31 2023

Copyright 2023 by Gentian Zavalani.
"""
# Standard library imports:
import numpy as np

def chebpts2(nx, ny=None, D=None, kind=2):
    # Third argument should be a domain.
    if D is not None:
        D = np.array(D).flatten()
        if D.size != 4:
            raise ValueError('Unrecognized domain.')

    # Default to the canonical domain [-1, 1, -1, 1] if not provided.
    if D is None:
        D = np.array([-1, 1, -1, 1])

    # Make it a square Chebyshev grid if only one input.
    if ny is None:
        ny = nx

    # Default to Chebyshev point of the 2nd kind.
    if kind is None:
        kind = 2

    # Get points.
    x = np.polynomial.chebyshev.chebpts2(nx)
    y = np.polynomial.chebyshev.chebpts2(ny)

    # Tensor product.
    xx, yy = np.meshgrid(x, y)

    return xx, yy