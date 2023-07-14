"""A class to instantiate a robot model and provides lots of methods through roboticstoolbox superclass
"""
import pathlib
import typing as tp
import tempfile
import trimesh
from trimesh.sample import sample_surface_even
import numpy as np
import random
from math import atan2, sqrt, sin, cos
from tqdm import tqdm
import matplotlib.pyplot as plt

from yourdfpy import URDF
from roboticstoolbox.robot.ERobot import ERobot
from sklearn.gaussian_process import GaussianProcessRegressor


class RobotModel(ERobot):
    """Child of ERobot, really just here to give access to all the methods of RTB with any URDF
    """

    @staticmethod
    def cart_to_pol(cart_coords) -> np.ndarray:

        nb_pts = cart_coords.shape[0]

        if cart_coords.shape[1] != 3:
            raise ValueError("Cartesian coordinates are not 3d")

        pol_coords = np.full([nb_pts, 3], np.nan)

        for i in range(nb_pts):
            (x, y, z) = cart_coords[i]
            pol_coords[i, 0] = atan2(y, x)  # Theta
            pol_coords[i, 1] = atan2(sqrt(x**2 + y**2), z)  # phi
            pol_coords[i, 2] = sqrt(x**2 + y**2 + z**2)  # rho

        return pol_coords

    @staticmethod
    def pol_to_cart(pol_coords) -> np.ndarray:

        nb_pts = pol_coords.shape[0]

        if pol_coords.shape[1] != 3:
            raise ValueError("Polar coordinates are not 3d")

        cart_coords = np.full([nb_pts, 3], np.nan)

        for i in range(nb_pts):
            (theta, phi, rho) = pol_coords[i]
            cart_coords[i, 0] = rho * sin(phi) * cos(theta)  # x
            cart_coords[i, 1] = rho * sin(phi) * sin(theta)  # y
            cart_coords[i, 2] = rho * cos(phi)  # z

        return cart_coords

    def __init__(self, description_folder: tp.Union[str, pathlib.Path],
                 description_file: tp.Union[str, pathlib.Path]) -> None:

        # For reproducibility
        random.seed(0)

        # Handle typing and path resolution
        if isinstance(description_folder, str):
            description_folder = pathlib.Path(description_folder)
        if isinstance(description_file, str):
            description_file = pathlib.Path(description_file)
        description_folder = description_folder.resolve()
        description_file = description_file.resolve()

        (e_links, name, self._urdf_str, self._file_path) = ERobot.URDF_read(description_file, tld=description_folder)
        super().__init__(e_links, name=name)

        # Normalise URDF string
        self._urdf_str = self._urdf_str.replace("package:/", str(description_folder))

        # Load geometry meshes
        self._visu_urdf = None
        self._visual_meshes = {}
        self._load_visual_urdf()
        self._load_visual_meshes()

        # Learned geometries
        self._learned_geometries = {}

    def _load_visual_urdf(self) -> None:

        # Load the file with yourdfpy
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_urdf = pathlib.Path(tmpdirname) / "tmp_urdf.urdf"
            with open(tmp_urdf, "w") as tmpf:
                tmpf.write(self._urdf_str)
            self._visu_urdf = URDF.load(tmp_urdf)

    def _load_visual_meshes(self) -> None:
        if self._visu_urdf is not None:
            for key, link in self._visu_urdf.link_map.items():
                if len(link.visuals) > 0:
                    stl_file = link.visuals[0].geometry.mesh.filename
                    self._visual_meshes[key] = trimesh.load_mesh(stl_file)
                    self._visual_meshes[key] = self._visual_meshes[key].process()
                    self._visual_meshes[key] = self._visual_meshes[key].smoothed()
        else:
            print("Can't load visual meshes before loading visual URDF")

    def learn_geometry(self, nb_learning_pts: int = 5000, verbose: bool = False) -> None:

        print("Learning geometries")
        training_scores = []
        for key, mesh in tqdm(self._visual_meshes.items()):
            cart_pts, _ = sample_surface_even(mesh, nb_learning_pts, seed=0)

            # Switch from tuple to array and certer mesh
            cart_pts_np = np.full((nb_learning_pts, 3), np.nan)
            for i in range(nb_learning_pts):
                cart_pts_np[i, 0] = cart_pts[i][0] - mesh.center_mass[0]
                cart_pts_np[i, 1] = cart_pts[i][1] - mesh.center_mass[1]
                cart_pts_np[i, 2] = cart_pts[i][2] - mesh.center_mass[2]
            
            # From cart to pol
            pol_vertices = RobotModel.cart_to_pol(cart_pts_np)
            X = pol_vertices[:, 0:2]
            y = np.squeeze(pol_vertices[:, 2])
            
            if key == "iiwa_link_0":
                fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                ax.scatter(np.squeeze(pol_vertices[:, 0]), np.squeeze(pol_vertices[:, 1]), np.squeeze(pol_vertices[:, 2]))
                fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                ax.scatter(np.squeeze(cart_pts_np[:, 0]), np.squeeze(cart_pts_np[:, 1]), np.squeeze(cart_pts_np[:, 2]))
            
            # Learn the geometry
            self._learned_geometries[key] = GaussianProcessRegressor().fit(X, y)

            if verbose:
                training_scores.append(self._learned_geometries[key].score(X, y))

        if verbose:
            for i, key in enumerate(self._learned_geometries.keys()):
                print(f"Training score for {key}: {training_scores[i]}")
            
    def show_geometries(self) -> None:
        for key in self._learned_geometries.keys():
            self.show_link_geometry(key)
        
        plt.show()
    
    def show_link_geometry(self, link_name: str, angluar_resolution: float = 0.05) -> None:
        
        # Bit of formatting in spherical coordinates
        theta = np.arange(-np.pi, np.pi, angluar_resolution)
        phi = np.arange(0, np.pi, angluar_resolution)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        theta_phi = np.array([theta_grid.flatten(), phi_grid.flatten()]).transpose()
    
        # Predict rho using model
        rho = self._learned_geometries[link_name].predict(theta_phi)
        
        # Switch to cartesian
        pol_coord = np.concatenate((theta_phi, np.expand_dims(rho, axis=1)), axis=1)
        cart_coord = RobotModel.pol_to_cart(pol_coord)
        
        # Format grid for plot
        X = np.squeeze(cart_coord[:, 0])
        Y = np.squeeze(cart_coord[:, 1])
        Z = np.squeeze(cart_coord[:, 2])
        
        # Plot surface
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.scatter(X, Y, Z)
        
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.scatter(np.squeeze(pol_coord[:, 0]), np.squeeze(pol_coord[:, 1]), np.squeeze(pol_coord[:, 2]))
        
