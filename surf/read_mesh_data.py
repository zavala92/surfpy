"""
Created on Wed Dec  1 08:22:31 2023

Copyright 2023 by Gentian Zavalani.
"""
# Standard library imports:
import scipy.io
import os

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