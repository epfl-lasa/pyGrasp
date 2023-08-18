import numpy as np
import trimesh
import pandas as pd
import pathlib
import pickle
import matplotlib.pyplot as plt
from itertools import combinations
import typing as tp

from .robot_model import RobotModel
from .reachable_spaces import ReachableSpace


class OppositionSpace(ReachableSpace):

    def __init__(self, robot_model: RobotModel) -> None:

        self.os_df = None

        super().__init__(robot_model)
        super().compute_rs()
        
        link_names = [link_name for link_name in self._rs_map.keys()]
        self.os_set = set(combinations(link_names, 2))

    def compute_os(self, force_recompute: bool = True) -> None:
        
        os_file_name = pathlib.Path(self.robot_model.name + "_os" + ".pickle")
        # Computing OS
        if not os_file_name.is_file() or force_recompute:
            link_names = [link_name for link_name in self._rs_map.keys()]

            self.os_df = pd.DataFrame(index=link_names, columns=link_names, dtype=object)
            self.os_dist = pd.DataFrame(index=link_names, columns=link_names, dtype=object)
            
            for base, target in self.os_set:
                if base != target:
                    if type(self.os_df.loc[base, target]) == float and np.isnan(self.os_df.loc[base, target]):
                        # Compute geometric OS
                        print(f"Computing os between {base} and {target}")
                        
                        # TODO: if we fix the definition of OS we can do a lot better here
                        self.os_df.loc[base, target] = trimesh.boolean.union([self._rs_map[base],
                                                                              self._rs_map[target]],
                                                                              engine='blender')
                        self.os_df.loc[base, target] = self.os_df.loc[base, target].convex_hull
                        self.os_df.loc[target, base] = self.os_df.loc[base, target]

                        # Get min and max OS
                        # TODO: This is not perfect but close enough 
                        distances = trimesh.proximity.signed_distance(self._rs_map[base], self._rs_map[target].vertices)
                        min_dst = distances.min()
                        max_dst = distances.max()
                        self.os_dist.at[base, target] = {'min': min_dst, 'max': max_dst}
                        self.os_dist.at[target, base] = self.os_dist.at[base, target]
            
            # Save the computed OS
            with open(str(os_file_name), 'wb') as fp:
                pickle.dump([self.os_df, self.os_dist], fp, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Loading os
        else:
            print("Loading OS's")
            with open(str(os_file_name), 'rb') as fp:
                os_data = pickle.load(fp)
                self.os_df = os_data[0]
                self.os_dist = os_data[1]

    def show_os(self, link_1: str, link_2: str, plot_rs: bool = False) -> tp.Optional[trimesh.Scene]:
        
        color_os = [255, 50, 150, 200]
        color_rs_1 = [255, 150, 50, 100]
        color_rs_2 = [205, 120, 40, 100]
        mesh_os = self.os_df.loc[link_1, link_2]
        mesh_rs_1 = self._rs_map[link_1]
        mesh_rs_2 = self._rs_map[link_2]
        robot_scene = None
        if type(mesh_os) == trimesh.Trimesh:
            robot_scene = self.robot_model.plot_robot(alpha=255)
            mesh_os.visual.face_colors = np.array([color_os] * mesh_os.faces.shape[0])
            mesh_rs_1.visual.face_colors = np.array([color_rs_1] * mesh_rs_1.faces.shape[0])
            mesh_rs_2.visual.face_colors = np.array([color_rs_2] * mesh_rs_2.faces.shape[0])
            robot_scene.add_geometry(mesh_os)
            if plot_rs:
                robot_scene.add_geometry([mesh_rs_1, mesh_rs_2])
        else:
            print(f"Empty Opposition space for {link_1} - {link_2}")
        
        return robot_scene
            
    def show_all_os(self) -> None:
        
        for link_1, link_2 in self.os_set:
            os_scene = self.show_os(link_1, link_2)
            if os_scene is not None:
                os_scene.show(caption=f"OS between {link_1} and {link_2}")

    def show_os_matrix(self, obj_diameter: float = 0) -> None:
        
        link_names = [link_name for link_name in self._rs_map.keys()]
        nb_links = len(link_names)
        
        # Create OS matrix
        os_dst_array = np.zeros([nb_links, nb_links])
        for i, link_1 in enumerate(link_names):
            for j, link_2 in enumerate(link_names):
                if link_1 != link_2:
                    os_dst_array[i, j] = self.os_dist.loc[link_1, link_2]['max']
                    if self.os_dist.loc[link_1, link_2]['min'] > obj_diameter > self.os_dist.loc[link_1, link_2]['max']:
                        os_dst_array[i, j] = 0
                        
        # Plot it
        fig, ax = plt.subplots()
        ax.imshow(os_dst_array)
        ax.set_xticks(np.arange(nb_links), labels=link_names)
        ax.set_yticks(np.arange(nb_links), labels=link_names)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        for i in range(nb_links):
            for j in range(nb_links):
                ax.text(j, i, os_dst_array[i, j], ha="center", va="center", color="w")
        
        ax.set_title(f"Oppositon spaces for d={obj_diameter}")
        fig.tight_layout()
        plt.show()
    
    @staticmethod
    def get_point_cloud_diameter(point_cloud) -> float:
        point_cloud = np.asarray(point_cloud)
        _, radius, _ = trimesh.nsphere(point_cloud)
        return radius * 2
        
    def get_best_os(self,
                    point_cloud: tp.Optional[np.ndarray] = None,
                    obj_diameter: tp.Optional[float] = None,
                    excluded_links: tp.List[str] = []) -> tp.Optional[tp.Tuple[str]]:
        best_os_combination = None
        max_os_dst = -np.inf
        
        if point_cloud is not None:
            obj_diameter = self.get_point_cloud_diameter(point_cloud)
        elif obj_diameter is None:
            raise ValueError("Either point_cloud or obj_diameter should not be None")
        
        for link_1, link_2 in self.os_set:
            if (link_1 not in excluded_links) and (link_2 not in excluded_links):
                if self.os_dist.loc[link_1, link_2]['min'] < obj_diameter < self.os_dist.loc[link_1, link_2]['max']:
                    if self.os_dist[link_1, link_2]['max'] > max_os_dst:
                        best_os_combination = (link_1, link_2)
                        max_os_dst = self.os_dist[link_1, link_2]['max']
        
        return best_os_combination
                
                