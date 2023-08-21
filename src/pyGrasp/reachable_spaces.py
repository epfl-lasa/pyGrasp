from dataclasses import dataclass
import typing as tp
import numpy as np
from tqdm import tqdm
from trimesh.boolean import union
import trimesh
import pathlib
import pickle
import pandas as pd

from . import utils as pgu
from .tools.alphashape import alphashape, circumradius
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

    MAX_ALPHA_FACES = 2000
    BASE_ALPHA_FACES = 500
    
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
        self._rs_map = {}
        # self._rs_df = 
    
    def compute_rs(self, angle_step: float = .1, force_recompute: bool = False) -> None:

        # Generate link map
        if not self._link_map:
            self._create_link_map()

        # Generate angle list
        angle_list = [np.arange(q_min, q_max, angle_step) for (q_min, q_max) in self.robot_model.qlim.transpose()]
        
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
                rs_file = pathlib.Path(self.robot_model.name + "_" + current_link_info.name + "_rs" + ".pickle")
                
                # Load a file if we find one
                if rs_file.is_file() and not force_recompute and self.robot_model.link_has_visual(current_link_info.name):
                    print(f"Loading for link: {current_link_info.name} ({nb_link_solved+1}/{len(self._link_map)})")
                    with open(str(rs_file), 'rb') as fp:
                        self._rs_map[current_link_info.name] = pickle.load(fp)
                    current_link_info.rs_solved = True
                        
                # Recompute the RS
                else:
                    force_recompute = True
                    print(f"Solving for link: {current_link_info.name} ({nb_link_solved+1}/{len(self._link_map)})")
                    self._solve_link(current_link_info, angle_list)
                    self._propagate_link(current_link_info, angle_list)
                    
                current_link_info = base_link_info   # TODO: Double chained link would allow us to be faster here
                nb_link_solved += 1
        
        if force_recompute:
            self._save_rs()
            
    def show_rs(self, link_name: str) -> trimesh.Scene:
        color = [255, 150, 50, 200]
        mesh = self._rs_map[link_name]
        mesh.visual.face_colors = np.array([color] * mesh.faces.shape[0])
        robot_scene = self.robot_model.plot_robot(alpha=255)
        robot_scene.add_geometry(mesh)
        return robot_scene

    def show_all_rs(self) -> None:

        for link_name in self._rs_map.keys():
            rs_scene = self.show_rs(link_name)
            rs_scene.show()

    def _solve_link(self, link_info: LinkInfo, angle_list: tp.List[np.ndarray]) -> None:
        
        q_test = self.robot_model.qz()

        # Iterate over all angles
        if link_info.joint_id is not None:
            nb_vertices = None
            vertices_list = None
            for i, q_moving in enumerate(tqdm(angle_list[link_info.joint_id])):
                
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

            # Compute alpha shape
            self._rs_map[link_info.name] = alphashape(vertices_list, link_info.alpha)
            
        # Link has no joint that directly moves it. So RS is link geometry
        elif self.robot_model.link_has_visual(link_info.name):     # TODO: Can't really think straight on how to handle the base link with no visual properly. TBD
            new_mesh = self.robot_model.get_mesh_at_q(None, link_info.name)
            max_alpha = self._find_max_alpha(new_mesh)
            self._rs_map[link_info.name] = alphashape(new_mesh.vertices, max_alpha)
        
        link_info.rs_solved = True

    def _propagate_link(self, parent_link_info, angle_list) -> None:
        """Propagating link deformation to all children of the current link
        """

        # Skip if no joint is directly connected to parent link. We'll get it later
        if parent_link_info.joint_id is not None:
            
            # Get all children of link
            link_children = self._get_link_children(parent_link_info.name)

            for next_link_name in link_children:

                # Get link to propagate to
                next_link_info = self._link_map[next_link_name]
                
                # Decimate RS if we estimate it is too big for propagation
                if self._rs_map[next_link_info.name].faces.shape[0] > self.MAX_ALPHA_FACES:
                    self._rs_map[next_link_info.name] = self._rs_map[next_link_info.name].simplify_quadric_decimation(self.BASE_ALPHA_FACES)

                # Setup vertices list
                nb_vertices = self._rs_map[next_link_info.name].vertices.shape[0]
                vertices_list = np.full([nb_vertices * len(angle_list[parent_link_info.joint_id]), 3], np.nan)

                # Iterate over all angles
                print(f"Propagating from {parent_link_info.name} to {next_link_info.name}")
                q_test = self.robot_model.qz()
                rs_backtransform = np.linalg.inv(self.robot_model.fkine(q_test, end=next_link_info.name))

                for i, q_moving in tqdm(enumerate(angle_list[parent_link_info.joint_id])):
                    q_test[parent_link_info.joint_id] = q_moving
                    
                    # Compute the new RS for the current link
                    fk = self.robot_model.fkine(q_test, end=next_link_info.name)
                    new_rs = self._rs_map[next_link_info.name].copy().apply_transform(rs_backtransform).apply_transform(fk)
                    
                    # Add vertices to list
                    vertices_list[i * nb_vertices:(i+1) * nb_vertices, :] = new_rs.vertices
                
                self._rs_map[next_link_info.name] = alphashape(vertices_list, self._find_max_alpha(self._rs_map[next_link_info.name]))
        
    def _create_link_map(self) -> None:
        """Generate a dictionary of links and their children to be able to brows them from base to tip
        """
        for i, link in enumerate(self.robot_model.links):
            
            # Only take into accounts links that have a visual geometry
            if not self.robot_model.link_has_visual(link.name):
                continue
            
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
            if link_info.parent_name is not None and link_info.parent_name in self._rs_map.keys():
                link_info.parent_id = self._rs_map[link_info.parent_name].id

    def _get_link_children(self, link_name: str) -> tp.List[str]:

        parent_link_info = self._link_map[link_name]

        children_list = parent_link_info.children_names

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
    
    def _save_rs(self) -> None:
        for key, rs in self._rs_map.items():
            rs_file = pathlib.Path(self.robot_model.name + "_" + key + "_rs" + ".pickle")
            
            with open(str(rs_file), 'wb') as fp:
                pickle.dump(rs, fp, protocol=pickle.HIGHEST_PROTOCOL)