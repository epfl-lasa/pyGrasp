"""A class to instantiate a robot model and provides lots of methods through roboticstoolbox superclass
"""
import pathlib
import pickle
import tempfile
import typing as tp
from math import cos, sin
from colorama import Fore, Style

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from roboticstoolbox.robot.ERobot import ERobot
from sklearn.gaussian_process import GaussianProcessRegressor
from tqdm import tqdm
from trimesh.sample import sample_surface_even
from yourdfpy import URDF

from . import utils as pgu
from .tools.hole_filling import fill_trimesh_holes

from JSDF import iiwa_JSDF


class RobotModel(ERobot):
    """Child of ERobot, really just here to give access to all the methods of RTB with any URDF
    """
    MIN_DECIMATION_FACES = 200

    @staticmethod
    def sanitize_mesh(mesh: trimesh.Trimesh) -> None:

        mesh.process()
        mesh.update_faces(mesh.unique_faces())
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
        mesh.merge_vertices()
        mesh.fix_normals()
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fix_inversion(mesh)

    @staticmethod
    def robust_hole_filling(msh: trimesh.Trimesh, rec_level: int = 0) -> trimesh.Trimesh:
        # Hole filling is hard. Trimesh sucks at this altohugh it shouldn't be that complicated.
        # Pyvista is slightly better... this works although it is not perfect

        # Try standard hole filling
        if msh.fill_holes():
            return msh

        # If it fails, identify broken faces and remove them until there is a clean hole in the mesh
        while len(msh.faces) > 0:

            # Remove broken faces
            brk = trimesh.repair.broken_faces(msh)
            brk_bool = np.full([msh.faces.shape[0]], True, dtype=bool)
            brk_bool[brk] = False
            msh.update_faces(brk_bool)

            # Check if we made a valid hole in the mesh
            faces = np.arange(len(msh.faces))
            hole = trimesh.path.exchange.misc.faces_to_path(msh, faces)['entities']
            hole = [e.points for e in hole if len(e.points) > 3 and e.points[0] == e.points[-1]]
            if len(hole) > 0:
                break

        # Couldn't fix the mesh so we give up
        else:
            return msh

        # Use pyvista for a better hole filling
        wt_msh = fill_trimesh_holes(msh, verbose=False)

        return wt_msh

    @staticmethod
    def cart_to_pol(cart_coords) -> np.ndarray:

        nb_pts = cart_coords.shape[0]

        if cart_coords.shape[1] != 3:
            raise ValueError("Cartesian coordinates are not 3d")

        pol_coords = np.full([nb_pts, 3], np.nan)

        x, y, z = cart_coords.T
        pol_coords[:, 0] = np.arctan2(y, x)  # Theta
        pol_coords[:, 1] = np.arctan2(np.sqrt(x**2 + y**2), z)  # phi
        pol_coords[:, 2] = np.sqrt(x**2 + y**2 + z**2)  # rho

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

        # Handle typing and path resolution
        if isinstance(description_folder, str):
            description_folder = pathlib.Path(description_folder)
        if isinstance(description_file, str):
            description_file = pathlib.Path(description_file)
        description_folder = description_folder.resolve()
        description_file = description_file.resolve()

        # Create rtb robot model
        (e_links, name, self._urdf_str, self._file_path) = ERobot.URDF_read(description_file, tld=description_folder)
        super().__init__(e_links, name=name)
        self._define_qz()  # define a zero position

        # Normalize URDF string
        self._urdf_str = self._urdf_str.replace("package:/", str(description_folder))

        # Load geometry meshes
        self._visu_urdf = None
        self._visual_meshes = {}
        self._simple_visual_meshes = {}
        self._load_visual_urdf()
        self._load_visual_meshes()

        # Learned geometries
        self._learned_geometries = {}

        # Working with learned geometries
        if 'iiwa' in name.lower() and False:
            self._jsdf_model = iiwa_JSDF(grid_size=[110] * 3)
        else:
            print(f"No JSDF model currently implemented for {name}")
            self._jsdf_model = None

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
            current_nb_pts = nb_learning_pts
            # Check if the file exist
            save_file = pgu.CACHE_FOLDER / pathlib.Path(self.name + "_" + key + ".pickle")
            if force_recompute or not save_file.is_file():

                if verbose:
                    print(f"Computing geometry model for link {key}")

                cart_pts, _ = sample_surface_even(mesh, current_nb_pts, seed=0)
                if cart_pts.shape[0] < current_nb_pts:
                    current_nb_pts = cart_pts.shape[0]
                    print(Fore.LIGHTRED_EX + f"WARNING: Only {current_nb_pts} points were sampled for link {key}" + Style.RESET_ALL)

                # Switch from tuple to array and center mesh
                cart_pts_np = np.full((current_nb_pts, 3), np.nan)
                for i in range(current_nb_pts):
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
        _, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.scatter(X, Y, Z, c='#17becf')
        ax.plot_trisurf(self._visual_meshes[link_name].vertices[:, 0],
                        self._visual_meshes[link_name].vertices[:, 1],
                        self._visual_meshes[link_name].vertices[:, 2],
                        triangles=self._visual_meshes[link_name].faces,
                        alpha=0.5)

    def link_param_to_abs(self,
                          link_name: str,
                          theta: float,
                          phi: float,
                          q: tp.List[float],
                          base_link: tp.Optional[str] = None) -> np.ndarray:

        cart_coords_local = self.link_param_to_link_abs(link_name, theta, phi)

        if base_link is None:
            base_link = self.base_link

        # Perform forward kinematic to link
        fk_result = self.fkine(q, link_name, base_link)

        # Pass cart mesh to absolute
        cart_mesh_coord_abs = np.dot(fk_result, np.concatenate((cart_coords_local, np.array([[1]])), axis=1).transpose())[0:3]

        return cart_mesh_coord_abs

    def link_param_to_link_abs(self, link_name: str, theta: float, phi: float) -> np.ndarray:

        rho = self._learned_geometries[link_name].predict(np.array([[theta, phi]]))
        sphe_coords = np.array([[theta, phi, rho[0]]])
        cart_coords = RobotModel.pol_to_cart(sphe_coords)
        cart_coords = cart_coords + self._visual_meshes[link_name].center_mass
        return cart_coords

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

        # Get point from learned geometry
        cart_mesh_coord = self.link_param_to_link_abs(tip_link.name, mesh_theta, mesh_phi)

        # Pass cart mesh to absolute
        cart_mesh_coord_abs = np.dot(fk_result, np.concatenate((cart_mesh_coord, np.array([[1]])), axis=1).transpose())[0:3]

        if plot_result:
            origin = np.zeros_like(cart_mesh_coord_abs)
            robot_scene = self.plot_robot(q)
            fk_len = np.linalg.norm(cart_mesh_coord_abs)
            fk_axis = trimesh.creation.cylinder(radius=0.001 * fk_len, sections=40, segment=np.concatenate([origin, cart_mesh_coord_abs], axis=1).transpose())
            fk_axis.visual.face_colors=[255, 0, 0, 255]
            robot_scene.add_geometry(fk_axis)
            robot_scene.show()

        return cart_mesh_coord_abs

    def plot_robot(self, q: tp.Optional[np.ndarray] = None, alpha: int = 200) -> trimesh.Scene:

        if q is None:
            q = self.qz()

        geom_list = []
        colors = [[200, 200, 200, alpha],
                  [100, 100, 100, alpha]]

        i = 0
        for key, mesh in self._simple_visual_meshes.items():

            fkine = self.fkine(q, end=key, start=self.base_link)
            transformed_mesh = mesh.copy().apply_transform(fkine)
            transformed_mesh.visual.face_colors = np.array([colors[i % 2]] * transformed_mesh.faces.shape[0])
            geom_list.append(transformed_mesh)
            i += 1

        robot_scene = trimesh.Scene(geometry=geom_list)

        return robot_scene

    def qz(self) -> np.ndarray:
        """Shortcut to get qz form the robot model

        Returns:
            np.ndarray: q zero
        """
        if 'qz' not in self.configs:
            self._define_qz()

        # display qz

        return self.configs['qz'].copy()

    def get_mesh_at_q(self,  q: tp.Optional[np.ndarray], link_name: str, simplified_mesh: bool = True) -> trimesh.Trimesh:

        # Get correct mesh
        if simplified_mesh:
            mesh = self._simple_visual_meshes[link_name].copy()
        else:
            mesh = self._visual_meshes[link_name].copy()

        # Compute and apply transform
        if q is not None:
            fkine = self.fkine(q, end=link_name, start=self.base_link)
            mesh = mesh.apply_transform(fkine)

        return mesh

    def link_has_visual(self, link_name: str) -> bool:
        return (len(self._visu_urdf.link_map[link_name].visuals) > 0)

    def check_collisions(self, q: tp.List[float], object: tp.Optional[trimesh.Trimesh]) -> float:

        cm = trimesh.collision.CollisionManager()

        for link_name in self._simple_visual_meshes.keys():
            cm.add_object(link_name, self.get_mesh_at_q(q, link_name))

        if object is not None:
            cm.add_object('object', object)

        is_collision, names, data_ls = cm.in_collision_internal(return_data=True, return_names=True)

        dst = 0
        if is_collision:
            for name, collision_data in zip(names, data_ls):
                # Only for non-consecutive links
                if 'object' not in name:
                    if (self.link_dict[name[0]].parent is not None and self.link_dict[name[0]].parent.name == name[1]) or \
                       (self.link_dict[name[1]].parent is not None and self.link_dict[name[1]].parent.name == name[0]):
                        continue
                dst = min(dst, -collision_data.depth)
        return dst

    def check_self_collisions(self, q: tp.List[float]) -> float:
        cm = trimesh.collision.CollisionManager()

        for link_name in self._simple_visual_meshes.keys():
            cm.add_object(link_name, self.get_mesh_at_q(q, link_name))

        is_collision, names, data_ls = cm.in_collision_internal(return_names=True, return_data=True)

        dst = 0
        if is_collision:
            for name, collision_data in zip(names, data_ls):

                # Only for non-consecutive links
                if (self.link_dict[name[0]].parent is not None and self.link_dict[name[0]].parent.name == name[1]) or \
                   (self.link_dict[name[1]].parent is not None and self.link_dict[name[1]].parent.name == name[0]):
                    continue

                dst = min(dst, -collision_data.depth)

        # TODO: extend distance function to non-collided space

        return dst

    def check_object_collisions(self, q: tp.List[float], object: trimesh.Trimesh) -> float:
        cm = trimesh.collision.CollisionManager()

        for link_name in self._simple_visual_meshes.keys():
            cm.add_object(link_name, self.get_mesh_at_q(q, link_name))

        is_collision, data_ls = cm.in_collision_single(object, return_data=True)

        if is_collision:
            dst = -max([data.depth for data in data_ls])
        else:
            dst = cm.min_distance_single(object)

        return dst

    def check_jsdf_collisions(self, q: tp.List[float], object: trimesh.Trimesh) -> np.ndarray:

        if self._jsdf_model is None:
            raise ValueError("No JSDF model for this robot type yet")

        self._jsdf_model.set_robot_joint_positions(q)

        signed_dst = -self._jsdf_model.calculate_signed_distance(object.vertices, min_dist=False)
        signed_dst = signed_dst.min(axis=0)

        return signed_dst

    def _load_visual_meshes(self, decimation_ratio: float = 0.1) -> None:
        if self._visu_urdf is not None:
            for key, link in self._visu_urdf.link_map.items():

                # Load full mesh
                if self.link_has_visual(key):
                    if link.visuals[0].geometry.mesh:
                        stl_file = link.visuals[0].geometry.mesh.filename
                        self._visual_meshes[key] = trimesh.load_mesh(stl_file)
                    elif link.visuals[0].geometry.box:
                        self._visual_meshes[key] = trimesh.creation.box(extents=link.visuals[0].geometry.box.size)
                    else:
                        raise ValueError(f"For now only box and mesh geometry are supported.\n Link has: {link.visual[0].geometry}")

                    # Scale link if needed
                    if link.visuals[0].geometry.mesh.scale is not None:
                        scale_matrix = np.identity(4)
                        scale_matrix[0, 0] = link.visuals[0].geometry.mesh.scale[0]
                        scale_matrix[1, 1] = link.visuals[0].geometry.mesh.scale[1]
                        scale_matrix[2, 2] = link.visuals[0].geometry.mesh.scale[1]
                        self._visual_meshes[key].apply_transform(scale_matrix)

                    # Set the link at it's origin point if needed
                    if link.visuals[0].origin is not None:
                        self._visual_meshes[key].apply_transform(link.visuals[0].origin)

                    # Load a simplified representation
                    if decimation_ratio < 1:
                        nb_faces = self._visual_meshes[key].faces.shape[0]
                        nb_decimated_faces = max([nb_faces * decimation_ratio, self.MIN_DECIMATION_FACES])
                        self._simple_visual_meshes[key] = self._visual_meshes[key].copy().simplify_quadric_decimation(nb_decimated_faces)

                        # Handle watertight
                        self.sanitize_mesh(self._simple_visual_meshes[key])
                    else:
                        raise ValueError(f"Decimation ratio should be in ]0, 1] but was {decimation_ratio}")
                    # breakpoint()
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

    def _define_qz(self) -> None:
        qz = np.zeros((self.n))

        # Adjust qz if not within joint limits
        for i in range(self.n):

            if qz[i] < self.qlim[0, i] or qz[i] > self.qlim[1, i]:
                qz[i] = self.qlim[0, i] + (self.qlim[1, i] - self.qlim[0, i])/2

        self.addconfiguration('qz', qz)
