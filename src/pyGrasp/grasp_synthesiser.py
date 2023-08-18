from scipy.optimize import minimize
import numpy as np
import typing as tp

from .robot_model import RobotModel
from .opposition_spaces import OppositionSpace

class GraspSynthesizer():
    
    def __init__(self, robot_model: RobotModel, force_recompute: bool = False) -> None:
        self._robot_model = robot_model
        self._robot_model.learn_geometry()
        self._os = OppositionSpace(robot_model)
        self._os.compute_os(force_recompute=force_recompute)
        
    def sythetize_grasp(self) -> None:
        
        for link_1, link_2 in self._os.os_set:
            self._synthtize_in_os(link_1, link_2)
    
    def _synthtize_in_os(self, link_1: str, link_2: str) -> None:
        """Grasp parameters
        - joint angles
        - coordinates of the contact points in 2D (2x2)  (theta [-pi-pi], phi [0, pi])
        """
        
        nb_params = self._robot_model.n + 2 * 2  # nb joints + 2 contact points in 2D

        # Initial point
        x0 = np.zeros([nb_params])
        
        # TODO: Truncate this better
        x0[0:self._robot_model.n] = self._robot_model.qz()
        
        # TODO: automatize this
        x0[self._robot_model.n] = 0             # theta 1
        x0[self._robot_model.n + 1] = np.pi / 2.  # phi 1
        x0[self._robot_model.n + 2] = 0         # theta 2
        x0[self._robot_model.n + 3] = np.pi / 2.  # phi 2
        
        # Bounds
        bounds = np.zeros([nb_params, 2])
        
        bounds[:self._robot_model.n] = self._robot_model.qlim.transpose()
        bounds[self._robot_model.n] = np.array[-np.pi, np.pi]
        bounds[self._robot_model.n + 1] = np.array[0, np.pi]
        bounds[self._robot_model.n + 2] = np.array[-np.pi, np.pi]
        bounds[self._robot_model.n + 3] = np.array[0, np.pi]
        
        # Constraints
        constraints = self._build_constraint_dict
        
        # Objective function
        obj_fun = lambda x: 1
        
        minimize(fun=obj_fun, x0=x0, bounds=bounds, constraints=constraints)
    
    def _get_active_joints_idx(self, link_contact_list: tp.List[str]) -> tp.List[int]:
        active_joints_idx = []
        
        for link_name in link_contact_list:
            contact_link_info = self._os._link_map[link_name]
            
            while True:
                if contact_link_info.joint_id and contact_link_info.joint_id not in active_joints_idx:
                    active_joints_idx.append(contact_link_info.joint_id)
                    
                if contact_link_info.parent_name is None:
                    break
                else:
                    contact_link_info = self._os._link_map[contact_link_info.parent_name]
    
        return active_joints_idx
        
        
    def _build_constraint_dict(self) -> tp.List[tp.Dict]:
        constraints = []
        return constraints
    
    def _create_object_center_constraint(self) -> tp.Dict:
        constraint = {}
        return constraint
        