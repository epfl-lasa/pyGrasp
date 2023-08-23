"""A file containing functions for hole filling"""

import pyvista as pv
import trimesh
import pymeshfix as mf


def pyvista_to_trimesh(pv_mesh: pv.PolyData) -> trimesh.Trimesh:
    """Convert a pyvista mesh to a trimesh mesh."""

    faces_as_array = pv_mesh.faces.reshape((pv_mesh.n_faces, 4))[:, 1:]
    return trimesh.Trimesh(vertices=pv_mesh.points, faces=faces_as_array)


def trimesh_to_pyvista(trimesh_mesh: trimesh.Trimesh) -> pv.PolyData:
    """Convert a trimesh mesh to a pyvista mesh."""

    nb_faces = trimesh_mesh.faces.shape[0]
    pv_faces_array = [0] * nb_faces * 4

    for i in range(nb_faces):
        pv_faces_array[i * 4] = 3  # Padding indicating number of points on the faces
        pv_faces_array[i * 4 + 1] = trimesh_mesh.faces[i, 0]
        pv_faces_array[i * 4 + 2] = trimesh_mesh.faces[i, 1]
        pv_faces_array[i * 4 + 3] = trimesh_mesh.faces[i, 2]

    return pv.PolyData(var_inp=trimesh_mesh.vertices, faces=pv_faces_array, n_faces=nb_faces)


def fill_trimesh_holes(mesh: trimesh.Trimesh, verbose: bool = False) -> trimesh.Trimesh:
    """
    Fill holes in a trimesh using meshfix.

    Parameters:
        mesh (trimesh.Trimesh): The trimesh object to fill the holes in.
        verbose (bool, optional): If True, print verbose output. Defaults to False.

    Returns:
        trimesh.Trimesh: The trimesh object with the holes filled.
    """

    pv_msh = trimesh_to_pyvista(mesh)
    meshfix = mf.MeshFix(pv_msh)
    meshfix.repair(verbose=verbose)
    repaired_mesh = pyvista_to_trimesh(meshfix.mesh)
    return repaired_mesh