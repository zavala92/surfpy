"""
Created on Wed Dec  1 08:22:31 2023

Copyright 2023 by Gentian Zavalani.
"""
# Standard library imports:
import numpy as np
from numba import njit


@njit(fastmath=True)
def SimpleImplicitSurfaceProjection(phi: callable, dphi: callable, x: np.ndarray, max_iter=10) -> np.ndarray:
    """Closest-point projection to surface given by implicit function 
    using a simple projection algorithm. Surface S
    is given by zero-levelset of function F. We assume that 
    F is differentiable in order to evaluate normals and to do an iterative projection.

    Parameters:
    ----------
    phi: zero-levelset function
    dphi: gradient of zero-levelset function
    x: the point to be projected

    Return
    ----------
    x: the projection point"""

    tol = 10 * np.finfo(np.float64).eps
    phi_v = phi(x)
    for i in range(max_iter):
        grad_phi = dphi(x)
        grad_phi_norm = np.sum(grad_phi ** 2)
        normalize = phi_v / grad_phi_norm

        if np.sqrt(phi_v * normalize) < tol:
            break

        for j in range(len(x)):
            x[j] -= grad_phi[j] * normalize

        phi_v = phi(x)

    return x