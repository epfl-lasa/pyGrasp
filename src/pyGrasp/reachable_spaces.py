from dataclasses import dataclass
import typing as tp
import numpy as np
from tqdm import tqdm
from trimesh.boolean import union
import trimesh
import matplotlib.pyplot as plt

from .tools.alphashape import alphashape, circumradius
from .robot_model import RobotModel


# TODO: The whole link map thing is a bit convoluted... maybe solve it at some point?

"""
 Here is what we do for a reachable spaces:

    
    # Resolve all links
    current_link = base_link
    while base_link not resolved:
        
        if current_link not resolved 
            if not current_link.all_descendence_resolved:
                current_link = furthest_unresolved_descendent
            else:
                compute current link reachable space as alpha shape
                propagate link movement on the children 
    
    
    # Resolve and propagate link
    for angle in joint_angle_list:
        move_robot to position with every angle to 0 but angle
        get mesh of current node
        get mesh of children at their place
        
                        
"""


@dataclass
class LinkInfo:
    name: str
    id: int
    rs_solved: bool
    joint_id: tp.Optional[int]
    children_id: list[int]
    children_names: list[str]
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
        self._rs_map = {}
    
    def compute(self, angle_step: float = .01, verbose=False) -> None:

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
                    
            if all_children_solved:
                print(f"Solving for link: {current_link_info.name} ({nb_link_solved+1}/{len(self._link_map)})")
                self._solve_link(current_link_info, angle_list)
                #self._propagate_link(current_link_info, angle_list)
                current_link_info = base_link_info   # TODO: Double chained link would allow us to be faster here
                nb_link_solved += 1
        
        breakpoint()
    
    def show_rs(self, link_name: str) -> None:
        
        _, ax = plt.subplots(subplot_kw={"projection": "3d"})
        
        mesh = self._rs_map[link_name]
        ax.plot_trisurf(mesh.vertices[:, 0],
                        mesh.vertices[:, 1],
                        mesh.vertices[:, 2],
                        triangles=mesh.faces, alpha=0.7)
        self.robot_model.plot_robot(alpha=1., ax=ax)
    
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
            
            msh = self._rs_map[link_info.name].copy()
            self.robot_model.sanitize_mesh(msh, fill_holes=False)
            
        # Link has no joint that directly moves it. So RS is link geometry
        elif self.robot_model.link_has_visual(link_info.name):     # TODO: Can't really think straight on how to handle the base link with no visual properly. TBD
            new_mesh = self.robot_model.get_mesh_at_q(None, link_info.name)
            max_alpha = self._find_max_alpha(new_mesh)
            self._rs_map[link_info.name] = alphashape(new_mesh.vertices, max_alpha)
        
        link_info.rs_solved = True

    def _propagate_link(self, parent_link_info, angle_list) -> None:
        """Propagating link deformation to all children of the current link
        """

        # Skip if no joint is directly connected to the parent link
        # TODO: That's a mistake we need to take into account multiple links without any actuators between them
        if parent_link_info.joint_id is not None:

            previous_link_info = parent_link_info

            while len(previous_link_info.children_names):

                # Get link to propagate to
                next_link_info = self._link_map[previous_link_info.children_names[0]]
                
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
                
                self._rs_map[next_link_info.name] = alphashape(vertices_list, next_link_info.alpha)
                
                # Move one link further
                previous_link_info = next_link_info
        
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
                                                     alpha=None)

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
                                                                alpha=None)
        
                # Append to children if parent already in dict
                else:
                    self._link_map[link.parent.name].children_id.append(i)
                    self._link_map[link.parent.name].children_names.append(link.name)
