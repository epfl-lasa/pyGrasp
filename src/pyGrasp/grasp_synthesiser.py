from scipy.optimize import minimize
import numpy as np
import typing as tp

from .robot_model import RobotModel
from .opposition_spaces import OppositionSpace

class GraspSynthesizer():

    NB_CONTACTS = 2
    ND_DIM_3D = 3

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

        # Get optimization parameters
        jnts = self._identify_relevant_joints(link_1, link_2)
        nb_params = len(jnts) + self.NB_CONTACTS * 2 + 3 + 4  # nb joints + contact points in 2D + object position + object orientation
        x0 = self._get_x0(jnts, nb_params)
        bnds = self._get_bounds(jnts, nb_params, link_1, link_2)
        constraints = self._build_constraint_dict()

        # Objective function
        obj_fun = lambda x: 1

        minimize(fun=obj_fun, x0=x0, bounds=bnds, constraints=constraints)

    def _get_x0(self, jnts: tp.List[int], nb_params: int) -> np.ndarray:

        nb_jnts = len(jnts)

        # Initial point
        x0 = np.zeros([nb_params])

        # TODO: Truncate this better
        x0[0:nb_jnts] = [self._robot_model.qz()[jnt_i] for jnt_i in jnts]

        # TODO: automatize this
        x0[nb_jnts] = 0             # theta 1
        x0[nb_jnts + 1] = np.pi / 2.  # phi 1
        x0[nb_jnts + 2] = 0         # theta 2
        x0[nb_jnts + 3] = np.pi / 2.  # phi 2
        x0[-1] = 1   # Unit quaternion for orientation
        return x0

    def _get_bounds(self, jnts: tp.List[int], nb_params: int, link_1: str, link_2: str) -> np.ndarray:
        nb_jnts = len(jnts)

        bounds = np.zeros([nb_params, 2])

        # Joint limitations
        bounds[:nb_jnts] = self._robot_model.qlim.transpose()[jnts]

        # Surface contact limitation
        bounds[nb_jnts] = [-np.pi, np.pi]
        bounds[nb_jnts + 1] = [0, np.pi]
        bounds[nb_jnts + 2] = [-np.pi, np.pi]
        bounds[nb_jnts + 3] = [0, np.pi]

        # Bounding box of the OS for object center
        os = self._os.os_df.loc[link_1, link_2]
        bounds[nb_jnts + self.NB_CONTACTS * 2: nb_jnts + self.NB_CONTACTS * 2 + self.ND_DIM_3D] = \
            os.bounds.transpose()

        # Some limitation for quaternions to ensure restrain space
        bounds[nb_jnts + self.NB_CONTACTS * 2 + self.ND_DIM_3D:] = np.array([[-1, 1]] * 4)

        return bounds

    def _identify_relevant_joints(self, link_1: str, link_2: str) -> tp.List[int]:

        # Get all parent joints
        jnts = self._os._find_all_parent_jnts(link_1)
        jnts += self._os._find_all_parent_jnts(link_2)

        # Remove multiple and sort
        jnts = list(set(jnts))
        jnts.sort()

        return jnts

    def _build_constraint_dict(self) -> tp.List[tp.Dict]:
        constraints = []
        return constraints

    def _create_object_center_constraint(self) -> tp.Dict:
        constraint = {}
        return constraint
