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
import pickle

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

    def learn_geometry(self, nb_learning_pts: int = 5000, force_recompute: bool = False, verbose: bool = False) -> None:
        """
        Learn the geometry of the visual meshes using GPR in spherical coordinates. 
        By default loads the model for an existing file if it exists. Else learns a new one and saves it.

        Args:
            nb_learning_pts (int, optional): The number of points to sample on the mesh for learning. Default: 5000.
            force_recompute (bool, optional): Whether to force recomputing the geometry model. Default: False.
            verbose (bool, optional): Whether to display verbose output. Defaults to False.

        Returns:
            None
        """

        print("Learning geometries")
        training_scores = []
        for key, mesh in tqdm(self._visual_meshes.items()):
            
            # Check if the file exist
            save_file = pathlib.Path(self.name + "_" + key + ".pickle")
            if force_recompute or not save_file.is_file():
                
                if verbose:
                    print(f"Computing geometry model for link {key}")
                
                cart_pts, _ = sample_surface_even(mesh, nb_learning_pts, seed=0)

                # Switch from tuple to array and center mesh
                cart_pts_np = np.full((nb_learning_pts, 3), np.nan)
                for i in range(nb_learning_pts):
                    cart_pts_np[i, 0] = cart_pts[i][0] - mesh.center_mass[0]
                    cart_pts_np[i, 1] = cart_pts[i][1] - mesh.center_mass[1]
                    cart_pts_np[i, 2] = cart_pts[i][2] - mesh.center_mass[2]
                
                # From cart to pol
                pol_vertices = RobotModel.cart_to_pol(cart_pts_np)
                X = pol_vertices[:, 0:2]
                y = np.squeeze(pol_vertices[:, 2])
                
                # Learn the geometry
                self._learned_geometries[key] = GaussianProcessRegressor().fit(X, y)

                if verbose:
                    training_scores.append(self._learned_geometries[key].score(X, y))
                
                # Save learned model
                with open(str(save_file), 'wb') as fp:
                    pickle.dump(self._learned_geometries[key], fp, protocol=pickle.HIGHEST_PROTOCOL)

            # Load saved files
            else:
                if verbose:
                    print(f"Found {save_file}, loading...")
                with open(str(save_file), 'rb') as fp:
                    self._learned_geometries[key] = pickle.load(fp)

        # Display results
        if verbose and len(training_scores) > 0:
            for i, key in enumerate(self._learned_geometries.keys()):
                print(f"Training score for {key}: {training_scores[i]}")
            
    def show_geometries(self) -> None:
        for key in self._learned_geometries.keys():
            self.show_link_geometry(key)
        
        plt.show()
    
    def show_link_geometry(self, link_name: str, angular_resolution: float = 0.1) -> None:
        
        # Bit of formatting in spherical coordinates
        theta = np.arange(-np.pi, np.pi, angular_resolution)
        phi = np.arange(0, np.pi, angular_resolution)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        theta_phi = np.array([theta_grid.flatten(), phi_grid.flatten()]).transpose()
    
        # Predict rho using model
        rho = self._learned_geometries[link_name].predict(theta_phi)
        
        # Switch to cartesian
        pol_coord = np.concatenate((theta_phi, np.expand_dims(rho, axis=1)), axis=1)
        cart_coord = RobotModel.pol_to_cart(pol_coord)
        
        # Format grid for plot
        X = np.squeeze(cart_coord[:, 0])+self._visual_meshes[link_name].center_mass[0]
        Y = np.squeeze(cart_coord[:, 1])+self._visual_meshes[link_name].center_mass[1]
        Z = np.squeeze(cart_coord[:, 2])+self._visual_meshes[link_name].center_mass[2]
        
        # Plot surface
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.scatter(X, Y, Z, c='#17becf')
        ax.plot_trisurf(self._visual_meshes[link_name].vertices[:, 0],
                        self._visual_meshes[link_name].vertices[:, 1],
                        self._visual_meshes[link_name].vertices[:, 2],
                        triangles=self._visual_meshes[link_name].faces,
                        alpha=0.5)      

    def extended_fk(self,
                    q: np.ndarray,
                    mesh_theta: float,
                    mesh_phi: float,
                    base_link=None,
                    tip_link=None,
                    plot_result: bool = False) -> np.ndarray:

        if base_link is None:
            base_link = self.base_link
            
        if tip_link is None:
            tip_link = self.links[-1]
        
        # Perform forward kinematic to link
        fk_result = self.fkine(q, tip_link, base_link)
        
        # Predict mesh rho
        mesh_rho = self._learned_geometries[tip_link.name].predict(np.array([[mesh_theta, mesh_phi]]))

        # Get mesh cartesian coordinates
        cart_mesh_coord = RobotModel.pol_to_cart(np.array([[mesh_theta, mesh_phi, mesh_rho[0]]]))
        
        # Center back mesh coord
        cart_mesh_coord = cart_mesh_coord + self._visual_meshes[tip_link.name].center_mass
        
        # Pass cart mesh to absolute
        cart_mesh_coord_abs = np.dot(fk_result, np.concatenate((cart_mesh_coord, np.array([[1]])), axis=1).transpose())[0:3]
        
        if plot_result:
            _, ax = plt.subplots(subplot_kw={"projection": "3d"})
            self.plot_robot(q, ax=ax)
            ax.quiver(0, 0, 0, [cart_mesh_coord_abs[0]], [cart_mesh_coord_abs[1]], [cart_mesh_coord_abs[2]], color='r', arrow_length_ratio=0.2, linewidths=2)
        plt.show()
        
        return cart_mesh_coord_abs
        
    def plot_robot(self, q: np.ndarray, ax=None) -> None:
        
        if ax is None:
            _, ax = plt.subplots(subplot_kw={"projection": "3d"})

        for key, mesh in self._visual_meshes.items():
            
            fkine = self.fkine(q, end=key, start=self.base_link)
    
            # Reduce mesh for smoother rendering
            simplified_mesh = mesh.simplify_quadric_decimation(0.01 * mesh.faces.shape[0]) 
            transformed_mesh = simplified_mesh.apply_transform(fkine)

            ax.plot_trisurf(transformed_mesh.vertices[:, 0],
                            transformed_mesh.vertices[:, 1],
                            transformed_mesh.vertices[:, 2],
                            triangles=transformed_mesh.faces, alpha=0.7)
        
        # Set axes limits
        xlim = ax.axes.get_xlim3d()
        ylim = ax.axes.get_ylim3d()
        zlim = ax.axes.get_zlim3d()

        lb = min([xlim[0], ylim[0], zlim[0]])
        ub = max([xlim[1], ylim[1], zlim[1]])

        ax.axes.set_xlim3d(lb, ub)
        ax.axes.set_ylim3d(lb, ub)
        ax.axes.set_zlim3d(lb, ub)

    def _load_visual_meshes(self) -> None:
        if self._visu_urdf is not None:
            for key, link in self._visu_urdf.link_map.items():
                if len(link.visuals) > 0:
                    stl_file = link.visuals[0].geometry.mesh.filename
                    mesh_origin = link.visuals[0].origin
                    self._visual_meshes[key] = trimesh.load_mesh(stl_file)
                    self._visual_meshes[key] = self._visual_meshes[key].process()
                    self._visual_meshes[key] = self._visual_meshes[key].smoothed()
                    self._visual_meshes[key] = self._visual_meshes[key].apply_transform(mesh_origin)
        else:
            print("Can't load visual meshes before loading visual URDF")    

    def _load_visual_urdf(self) -> None:
        """
        Loads a URDF to use late the visual meshes of it.
        """

        # Load the file with yourdfpy
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_urdf = pathlib.Path(tmpdirname) / "tmp_urdf.urdf"
            with open(tmp_urdf, "w") as tmpf:
                tmpf.write(self._urdf_str)
            self._visu_urdf = URDF.load(tmp_urdf)
        