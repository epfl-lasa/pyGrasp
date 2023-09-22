from dataclasses import dataclass
import typing as tp
import numpy as np
from tqdm import tqdm
import trimesh
import pathlib
import pickle
import pandas as pd

from . import utils as pgu
from .tools.alphashape import alphashape, circumradius, OPTIMAL_VERT_NUMBER
from .robot_model import RobotModel


@dataclass
class LinkInfo:
    name: str
    id: int
    rs_solved: bool
    joint_id: tp.Optional[int]
    children_id: list[int]
    children_names: list[str]
    parent_id: int
    parent_name: tp.Optional[str]
    alpha: tp.Optional[float]


class ReachableSpace:

    MAX_ALPHA_FACES = 5000
    BASE_ALPHA_FACES = 1000

    @staticmethod
    def _find_max_alpha(mesh: trimesh.Trimesh) -> float:
        """Finds the maximum alpha of a given set of vertices.
        The maximum alpha is calculated as the biggest alpha that still yields a single concurrent shape

        Args:
            vertices (np.ndarray): _description_

        Returns:
            float: _description_
        """

        # Iterate over all points to find the closest ones in the mesh
        max_circumradius = 0
        for vrtx_ids in mesh.faces:
            try:
                cr = circumradius(mesh.vertices[vrtx_ids])
                max_circumradius = max([max_circumradius, cr])
            except:
                print("Degenerated face")

        # Alpha is 1 over radius
        return 1/max_circumradius

    def __init__(self, robot_model: RobotModel) -> None:
        self.robot_model = robot_model
        self._link_map = {}   # Key = link, value = LinkInfo
        self._rs_df = None
        self._root_link = None

    @property
    def root_link(self) -> str:
        if self._root_link is None:
            self._root_link = self._find_valid_root_link()

        return self._root_link

    def compute_rs(self, angle_step: float = .1, force_recompute: bool = False) -> None:

        # Generate link map
        if not self._link_map:
            self._create_link_map()

        # Generate angle list
        angle_list = [np.arange(q_min, q_max, angle_step) for (q_min, q_max) in self.robot_model.qlim.transpose()]

        # Get all link names with a geometry
        all_links = self._get_link_children(self.root_link)
        all_links.append(self.root_link)
        valid_links = []
        for link_name in all_links:
            if self.robot_model.link_has_visual(link_name):
                valid_links.append(link_name)

        # Make a dataframe to hold all relative reachable spaces
        self._rs_df = pd.DataFrame(index=valid_links, columns=valid_links, dtype=object)

        # Check if there is a file to load
        rs_file = pgu.CACHE_FOLDER / pathlib.Path(self.robot_model.name + "_rs" + ".pickle")
        if rs_file.is_file() and not force_recompute:
            print("Loading RS file")
            with open(str(rs_file), 'rb') as fp:
                self._rs_df = pickle.load(fp)

        # Computing RS
        else:
            # Solve every link reachable space from tip to base
            base_link_info = self._link_map[self.robot_model.base_link.name]
            current_link_info = base_link_info
            nb_link_solved = 0
            while nb_link_solved < len(self._link_map):

                # Check for unsolved descendants (those have to be solved first)
                all_children_solved = True
                for child_id in current_link_info.children_id:
                    child_link_info = self._link_map[self.robot_model.links[child_id].name]
                    if not child_link_info.rs_solved:
                        current_link_info = child_link_info
                        all_children_solved = False
                        break

                # Solve the current link RS
                if all_children_solved:
                    force_recompute = True
                    print(f"Solving for link: {current_link_info.name} ({nb_link_solved+1}/{len(self._link_map)})")
                    self._solve_link(current_link_info, angle_list)
                    self._propagate_link(current_link_info, angle_list)

                    current_link_info = base_link_info   # TODO: Double chained link would allow us to be faster here
                    nb_link_solved += 1

            # Save the computed RS
            with open(str(rs_file), 'wb') as fp:
                pickle.dump(self._rs_df, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def show_rs(self, link_name: str, base_link: tp.Optional[str] = None) -> tp.Optional[trimesh.Scene]:
        color = [255, 150, 50, 200]
        robot_scene = None

        if base_link is None:
            base_link = self.root_link
        if not self.robot_model.link_has_visual(base_link) or not self.robot_model.link_has_visual(link_name):
            return robot_scene
        mesh = self._rs_df.loc[base_link, link_name]

        # Root link is handled differently
        if type(mesh) == float and np.isnan(mesh):
            mesh = self._rs_df.loc[base_link, link_name]

        # This shouldn't happen but an error would be annoying for debug
        if type(mesh) == float and np.isnan(mesh):
            print(f"Couldn't find RS from {base_link} to {link_name}")

        # Show the mesh
        else:
            mesh.visual.face_colors = np.array([color] * mesh.faces.shape[0])
            robot_scene = self.robot_model.plot_robot(alpha=255)
            robot_scene.add_geometry(mesh)

        return robot_scene

    def show_all_rs(self) -> None:

        for link_name in self._rs_df.columns:
            rs_scene = self.show_rs(link_name)
            if rs_scene is not None:
                rs_scene.show(caption=f"RS for {link_name}")

    def _solve_link(self, link_info: LinkInfo, angle_list: tp.List[np.ndarray]) -> None:
        # TODO: This method is optimized but became a bit messy. Clean up
        q_test = self.robot_model.qz()

        # Find nearest valid parent
        parent_link = None
        current_link = link_info
        while True:
            if current_link.parent_name is None:
                break
            else:
                if self.robot_model.link_has_visual(self._link_map[current_link.parent_name].name):
                    parent_link = self._link_map[current_link.parent_name]
                    break

                current_link = self._link_map[current_link.parent_name]

        # Fill link self geometry
        if self.robot_model.link_has_visual(link_info.name):
            new_mesh = self.robot_model.get_mesh_at_q(self.robot_model.qz(), link_info.name)
            if link_info.alpha is None:
                        link_info.alpha = self._find_max_alpha(new_mesh)
            self._rs_df.loc[link_info.name, link_info.name] = alphashape(new_mesh.vertices, link_info.alpha)
            # self._rs_df.loc[link_info.name, link_info.name] = \
            #     RobotModel.robust_hole_filling(self._rs_df.loc[link_info.name, link_info.name])

        # Iterate over all angles
        if link_info.joint_id is not None:

            nb_vertices = None
            vertices_list = None

            # Solve the link
            if parent_link is not None and self.robot_model.link_has_visual(link_info.name):
                i_bottom = 0
                preshaped_vert = np.zeros((0, 3))
                intermediary_ashape = None
                for i, q_moving in tqdm(enumerate(angle_list[link_info.joint_id])):

                    # Compute FK
                    q_test[link_info.joint_id] = q_moving
                    new_mesh = self.robot_model.get_mesh_at_q(q_test, link_info.name)

                    # Choose a relevant alpha
                    if nb_vertices is None:
                        nb_vertices = new_mesh.vertices.shape[0]
                        vertices_list = np.empty([nb_vertices * len(angle_list[link_info.joint_id]), 3])

                    if link_info.alpha is None:
                        link_info.alpha = self._find_max_alpha(new_mesh)

                    vertices_list[i * nb_vertices:(i+1) * nb_vertices, :] = new_mesh.vertices

                    # Perform alpha shape. If we get too far from the optimal number of vertices, it will take forever
                    if ((i + 1 - i_bottom) * nb_vertices) + preshaped_vert.shape[0] > OPTIMAL_VERT_NUMBER:
                        concat_vert = np.concatenate((preshaped_vert, vertices_list[i_bottom*nb_vertices:(i+1)*nb_vertices, :]), axis=0)
                        intermediary_ashape = alphashape(concat_vert, link_info.alpha)

                        # Simplify mesh to reasonable number of vertices if it is still too big.
                        if intermediary_ashape.vertices.shape[0] > 2*OPTIMAL_VERT_NUMBER:
                            intermediary_ashape = intermediary_ashape.simplify_quadric_decimation(0.5 * OPTIMAL_VERT_NUMBER)

                        preshaped_vert = intermediary_ashape.vertices
                        i_bottom = i+1

                # Compute alpha shape
                if i_bottom * nb_vertices < vertices_list.shape[0]:
                    vertices_list = np.concatenate((preshaped_vert, vertices_list[i_bottom*nb_vertices:, :]), axis=0)
                    self._rs_df.loc[parent_link.name, link_info.name] = \
                        alphashape(vertices_list, link_info.alpha)
                elif intermediary_ashape is not None:
                    self._rs_df.loc[parent_link.name, link_info.name] = intermediary_ashape
                else:
                    raise ValueError("This shouldn't happen, the code is faulty. Please report through git issues.")

        # Link has no joint that directly moves it. So RS is link geometry
        elif self.robot_model.link_has_visual(link_info.name):     # TODO: How to handle the links w/o visual properly.
            new_mesh = self.robot_model.get_mesh_at_q(self.robot_model.qz(), link_info.name)
            max_alpha = self._find_max_alpha(new_mesh)

            # This is done for the case the joint between link is just "static"
            if parent_link is not None:
                self._rs_df.loc[parent_link.name, link_info.name] = alphashape(new_mesh.vertices, max_alpha)

        if parent_link is not None and self.robot_model.link_has_visual(link_info.name) and self.robot_model.link_has_visual(parent_link.name) and self._rs_df.loc[parent_link.name, link_info.name].is_watertight:

            # Fun fact... I think that in certain cases, this thing yields a segfault.
            # No fault of mine, the library sucks.
            self._rs_df.loc[parent_link.name, link_info.name] = \
                RobotModel.robust_hole_filling(self._rs_df.loc[parent_link.name, link_info.name])

        link_info.rs_solved = True

    #TODO: That is a lot of duplicated code
    def _propagate_link(self, parent_link_info, angle_list) -> None:
        """Propagating link deformation to all children of the current link
        """

        # Skip if no joint is directly connected to parent link. We'll get it later
        if parent_link_info.joint_id is not None:

            # Get all children of link
            link_children = self._get_link_children(parent_link_info.name)

            # Get name of the closest stable link (not moving)
            # We're gonna register the geometry under this name in the DF
            parent_iterator = parent_link_info
            stable_link = None
            while parent_iterator.parent_name is not None:
                parent_iterator = self._link_map[parent_iterator.parent_name]
                if self.robot_model.link_has_visual(parent_iterator.name):
                    stable_link = parent_iterator
                    break

            # Handle errors
            if stable_link is None:
                return
                raise ValueError(f"Link {parent_link_info.name} has no parent with visual mesh")

            for next_link_name in link_children:

                # Get link to propagate to
                next_link_info = self._link_map[next_link_name]

                # Get link from which the propagation was done last:
                previous_parent = None
                parent_iterator = next_link_info
                while parent_iterator.parent_name is not None \
                        and parent_iterator.parent_name != stable_link.name:

                    parent_iterator = self._link_map[parent_iterator.parent_name]
                    if self.robot_model.link_has_visual(parent_iterator.name):
                        previous_parent = parent_iterator

                # Handle potential errors
                if previous_parent is None:  # This should never happen
                    return
                    # raise ValueError(f"Link {next_link_info.name} has no parent with visual but should")

                # This shouldn't happen either
                if type(self._rs_df.loc[previous_parent.name, next_link_name]) == float and \
                        np.isnan(self._rs_df.loc[previous_parent.name, next_link_name]):
                    raise ValueError("Trying to propagate from a link whose children have not all been solved")

                # Decimate RS if we estimate it is too big for propagation
                parent_mesh = self._rs_df.loc[previous_parent.name, next_link_name].copy()
                if self._rs_df.loc[previous_parent.name, next_link_name].faces.shape[0] > self.MAX_ALPHA_FACES:
                    parent_mesh = parent_mesh.simplify_quadric_decimation(self.BASE_ALPHA_FACES)

                # Setup vertices list
                nb_vertices = parent_mesh.vertices.shape[0]
                vertices_list = np.full([nb_vertices * len(angle_list[parent_link_info.joint_id]), 3], np.nan)

                # Iterate over all angles
                print(f"Propagating from {parent_link_info.name} to {next_link_info.name}")
                q_test = self.robot_model.qz()
                rs_backtransform = np.linalg.inv(self.robot_model.fkine(q_test, end=next_link_info.name))

                # Replace parent mesh at 0 origin
                parent_mesh.apply_transform(rs_backtransform)

                current_alpha = self._find_max_alpha(parent_mesh)

                # Used to incementally compute alphashape (for speed)
                i_bottom = 0
                preshaped_vert = np.zeros((0, 3))
                intermediary_ashape = None
                for i, q_moving in tqdm(enumerate(angle_list[parent_link_info.joint_id])):
                    q_test[parent_link_info.joint_id] = q_moving

                    # Compute the new RS for the current link
                    fk = self.robot_model.fkine(q_test, end=next_link_info.name)
                    new_rs = parent_mesh.copy().apply_transform(fk)

                    # Add vertices to list
                    vertices_list[i * nb_vertices:(i+1) * nb_vertices, :] = new_rs.vertices

                    # Perform alpha shape. If we get too far from the optimal number of vertices, it will take forever
                    if ((i + 1 - i_bottom) * nb_vertices) + preshaped_vert.shape[0] > OPTIMAL_VERT_NUMBER:
                        concat_vert = np.concatenate((preshaped_vert, vertices_list[i_bottom*nb_vertices:(i+1)*nb_vertices, :]), axis=0)
                        intermediary_ashape = alphashape(concat_vert, current_alpha)

                        # Simplify mesh to reasonable number of vertices if it is still too big.
                        if intermediary_ashape.vertices.shape[0] > 2*OPTIMAL_VERT_NUMBER:
                            intermediary_ashape = intermediary_ashape.simplify_quadric_decimation(0.5 * OPTIMAL_VERT_NUMBER)
                        preshaped_vert = intermediary_ashape.vertices
                        i_bottom = i+1

                if i_bottom * nb_vertices < vertices_list.shape[0]:
                    vertices_list = np.concatenate((preshaped_vert, vertices_list[i_bottom*nb_vertices:, :]), axis=0)
                    self._rs_df.loc[stable_link.name, next_link_name] = \
                        alphashape(vertices_list, current_alpha)
                elif intermediary_ashape is not None:
                    self._rs_df.loc[stable_link.name, next_link_name] = intermediary_ashape
                else:
                    raise ValueError("This shouldn't happen, the code is faulty. Please report through git issues.")

    def _create_link_map(self) -> None:
        """Generate a dictionary of links and their children to be able to brows them from base to tip
        """
        for i, link in enumerate(self.robot_model.links):

            # Only take into accounts links that have a visual geometry
            if not self.robot_model.link_has_visual(link.name):
                pass

            # Add link to the dictionary if needed
            if link.name not in self._link_map:
                self._link_map[link.name] = LinkInfo(name=link.name,
                                                     id=i,
                                                     rs_solved=False,
                                                     joint_id=link.jindex,
                                                     children_id=[],
                                                     children_names=[],
                                                     alpha=None,
                                                     parent_id=-1,
                                                     parent_name=link.parent if link.parent is None else link.parent.name)

            # Means we added it as a parent and couldn't fill the id and joint_id yet so we fix it now
            else:
                self._link_map[link.name].id = i
                self._link_map[link.name].joint_id = link.jindex

            # Add parent to the dictionary if needed
            if link.parent is not None:
                if link.parent.name not in self._link_map:
                    self._link_map[link.parent.name] = LinkInfo(name=link.parent.name,
                                                                id=-1,
                                                                rs_solved=False,
                                                                joint_id=None,
                                                                children_id=[i],
                                                                children_names=[link.name],
                                                                alpha=None,
                                                                parent_id=-1,
                                                                parent_name=link.parent.parent if link.parent.parent is None else link.parent.parent.name)

                # Append to children if parent already in dict
                else:
                    self._link_map[link.parent.name].children_id.append(i)
                    self._link_map[link.parent.name].children_names.append(link.name)

        # One more pass to fill all parent ids
        for link_info in self._link_map.values():
            if link_info.parent_name is not None and link_info.parent_name in self._link_map.keys():
                link_info.parent_id = self._link_map[link_info.parent_name].id

    def _get_link_children(self, link_name: str) -> tp.List[str]:

        parent_link_info = self._link_map[link_name]

        children_list = parent_link_info.children_names.copy()

        # This isn't super effective but makes for simpler code
        added_new_child = True
        while added_new_child:

            added_new_child = False
            for child_name in children_list:

                for grand_child_name in self._link_map[child_name].children_names:

                    if grand_child_name not in children_list:
                        children_list.append(grand_child_name)
                        added_new_child = True

        return children_list

    def _find_valid_root_link(self) -> str:

        root_link = None

        for current_link in self._link_map.values():

            while True:
                if self.robot_model.link_has_visual(current_link.name):
                    root_link = current_link
                else:
                    root_link = current_link

                parent_name = self._link_map[current_link.name].parent_name
                if parent_name is not None:
                    current_link = self._link_map[parent_name]
                else:
                    break

            # This works because we assume there can only be one root link
            if root_link is not None:
                break

        if root_link is None:
            raise ValueError("No root link with a geometry found")

        return root_link.name

    def _find_first_common_parent(self, link_1: str, link_2: str) -> tp.Optional[str]:

        if link_1 == link_2:
            return link_1

        link_info_1 = self._link_map[link_1]
        link_info_2 = self._link_map[link_2]

        # TODO: Redundant code. If we're clever we could get around this
        # TODO: This is also easy to extend to as many links as we want
        # Establish common parents for link 1
        link_1_parent_list = []
        link_1_parent = link_info_1
        while link_1_parent.parent_name is not None:

            link_1_parent = self._link_map[link_1_parent.parent_name]

            if self.robot_model.link_has_visual(link_1_parent.name):

                if link_1_parent.name == link_2:
                    return link_1_parent.name

                link_1_parent_list.append(link_1_parent.name)

        # Establish parents of link 2
        link_2_parent = link_info_2
        while link_2_parent.parent_name is not None:

            link_2_parent = self._link_map[link_2_parent.parent_name]

            if self.robot_model.link_has_visual(link_2_parent.name):

                if link_2_parent.name == link_1 or link_2_parent.name in link_1_parent_list:
                    return link_2_parent.name

        return None

    def _find_all_parent_jnts(self, link: str) -> tp.List[int]:

        parent_jnts_list = []

        while link is not None:

            link_info = self._link_map[link]

            if link_info.joint_id is not None:
                parent_jnts_list.append(link_info.joint_id)

            link = link_info.parent_name

        return parent_jnts_list

