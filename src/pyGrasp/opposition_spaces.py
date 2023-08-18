import numpy as np
import trimesh
import pandas as pd

from .robot_model import RobotModel
from .reachable_spaces import ReachableSpace

class OppositionSpace(ReachableSpace):

    def __init__(robot_model: RobotModel) -> None:

        self.os_df = None

        super().__init__(robot_model)
        super().compute_rs()

    def compute_os(self) -> None:
        
        link_names = [link_name for link_name in self._rs_map.keys()]

        self.os_df = pd.DataFrame(rows=link_names, columns=link_names, dtype=object)
        self.os_dist = pd.DataFrame(rows=link_names, columns=link_names, dtype=dict)
        
        for base in link_names:
            for target in link_names:

                if base != target:
                    if not self.os_df[base, target]:
                        # Compute geometric OS
                        self.os_df[base, target] = trimesh.boolean.intersection([self._rs_map[base],
                                                                                 self._rs_map[target]],
                                                                                 engine='blender')
                        self.os_df[target, base] = self.os_df[base, target]

                        # Get min and max OS
