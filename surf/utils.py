import numpy as np
import scipy.io
import os
from numba import njit

__all__ = ['transform', 'chebpts2','read_mesh_data','SimpleImplicitSurfaceProjection']

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

def read_mesh_data(mesh_path):
    """
    Read mesh data from a MAT file.

    Args:
        mesh_path (str): The file path to the MAT file containing mesh data.

    Returns:
        vertices (numpy.ndarray): The 'vertices' data from the MAT file.
        faces (numpy.ndarray): The 'faces' data from the MAT file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        Exception: If an error occurs during file reading.
    """
    try:
        # Check if the file exists
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"File not found: {mesh_path}")

        # Load the MAT file
        mesh_mat = scipy.io.loadmat(mesh_path)

        # Get a list of keys in the loaded dictionary
        key_list = list(mesh_mat.keys())

        # Access the 'vertices' and 'faces' data
        vertices = mesh_mat[key_list[-1]]
        faces = mesh_mat[key_list[-2]] - 1  

        return vertices, faces
    except Exception as e:
        print(f"An error occurred while reading the mesh data: {e}")
        return None, None
@njit(fastmath=True)
def SimpleImplicitSurfaceProjection(phi: callable, dphi: callable, x: np.ndarray, max_iter=10) -> np.ndarray:
    """Closest-point projection to surface given by implicit function 
    using a simple projection algorithm.Surface S
    is given by zero-levelset of function F. We assume that 
    F is differentiable in order to evaluate normals and tomdo an iterative projection.

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
        grad_phi_norm = np.sum(grad_phi**2)
        normalize = phi_v / grad_phi_norm

        if np.sqrt(phi_v * normalize) < tol:
            break

        for j in range(len(x)):
            x[j] -= grad_phi[j] * normalize

        phi_v = phi(x)

    return x