from scipy.optimize import minimize
import numpy as np
import typing as tp
import trimesh
from scipy.spatial.transform import Rotation
from scipy.optimize import OptimizeResult
import time

from .robot_model import RobotModel
from .opposition_spaces import OppositionSpace


class GraspSynthesizer():

    NB_CONTACTS = 2
    ND_DIM_3D = 3
    CONTACT_DST_TOL = 0

    def __init__(self, robot_model: RobotModel, force_recompute: bool = False) -> None:
        self._robot_model = robot_model
        self._robot_model.learn_geometry()
        self._os = OppositionSpace(robot_model)

        print("Computing opposition spaces...")
        self._os.compute_os(force_recompute=force_recompute)
        print("Opposition spaces computed.")

    def sythetize_grasp(self, object: trimesh.Trimesh) -> None:

        for link_1, link_2 in self._os.os_set:
            self.synthtize_in_os(link_1, link_2, object)

    def synthtize_in_os(self, link_1: str, link_2: str, object: trimesh.Trimesh) -> None:
        """Grasp parameters
        - joint angles
        - coordinates of the contact points in 2D (2x2)  (theta [-pi-pi], phi [0, pi])
        """
        # Get optimization parameters
        jnts = self._identify_relevant_joints(link_1, link_2)
        nb_params = len(jnts) + self.NB_CONTACTS * 2 + 3 + 4  # nb joints + contact points in 2D + object position + object orientation
        x0 = self._get_x0(jnts, nb_params)
        bnds = self._get_bounds(jnts, nb_params, link_1, link_2)
        constraints = self._build_constraint_dict(jnts, object, link_1, link_2)
        t1 = time.time()
        print(f"Starting optimization for OS {link_1}-{link_2}")
        opt_result = minimize(fun=lambda x: GraspSynthesizer.objective_function(x),
                              x0=x0,
                              bounds=bnds,
                              constraints=constraints)
        t2 = time.time()
        print(f"Optimization time: {t2-t1}")
        print(f"Optimization result: {opt_result}")
        self._show_optim_result(opt_result.x, jnts, object, [link_1, link_2])

    def _show_optim_result(self, x: tp.List[float], active_jnts: tp.List[int], object: trimesh.Trimesh, links: tp.List[str]) -> None:

        # Add robot to scene
        q = self._robot_model.qz()
        q[active_jnts] = x[:len(active_jnts)]
        scene = self._robot_model.plot_robot(q)

        # Add object to scene
        cyan = [0, 255, 255, 210]
        quat = x[len(active_jnts) + self.NB_CONTACTS * 2 + self.ND_DIM_3D:]
        trans = x[len(active_jnts) + self.NB_CONTACTS * 2: len(active_jnts) + self.NB_CONTACTS * 2 + self.ND_DIM_3D]
        obj_transform = GraspSynthesizer.trans_quat_to_mat(trans, quat)
        obj_inplace = object.copy().apply_transform(obj_transform)
        obj_inplace.visual.face_colors = np.array([cyan] * obj_inplace.faces.shape[0])
        scene.add_geometry(obj_inplace)

        # TODO: Add contact points
        red = [255, 100, 100, 255]
        contact_spheres = []
        for i in range(self.NB_CONTACTS):
            contact_spheres.append(trimesh.creation.icosphere(radius=0.005))
            contact_param = x[len(active_jnts)+2*i:len(active_jnts)+2*i+2]
            robot_contact_point = self._robot_model.link_param_to_abs(links[i], contact_param[0], contact_param[1], q)
            contact_transform = np.identity(4)
            contact_transform[:3, 3] = np.squeeze(robot_contact_point)
            contact_spheres[-1].apply_transform(contact_transform)
            contact_spheres[-1].visual.face_colors = np.array([red] * contact_spheres[-1].faces.shape[0])
            scene.add_geometry(contact_spheres[-1])

        scene.show()

    @staticmethod
    def trans_quat_to_mat(translation: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
        se3_transform = np.zeros((4, 4))
        rotation = Rotation.from_quat(quaternion)
        se3_transform[:3, :3] = rotation.as_matrix()
        se3_transform[3, 3] = 1
        se3_transform[:3, 3] = translation

        return se3_transform

    @staticmethod
    def objective_function(x: tp.List[float]) -> float:
        return 1

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

    def _build_constraint_dict(self,
                               active_joints: tp.List[int],
                               object: trimesh.Trimesh,
                               link_1: str, link_2: str) -> tp.List[tp.Dict]:
        nb_joints = len(active_joints)
        constraints = []

        constraints.append(self._quaternion_constraint(nb_joints))
        # constraints.append(self._global_collision_constraint(object, active_joints))
        # constraints.append(self._contact_constraint_on_learned_geom(object, active_joints, 0, link_1))
        constraints.append(self._contact_constraint_on_real_geom(object, active_joints, 1, link_2))
        constraints.append(self._self_collision_constraint(active_joints))
        constraints.append(self._robot_object_collision_constraint(object, active_joints))

        return constraints

    def _quaternion_constraint(self, nb_jnts: int) -> tp.Dict:
        constraint = {}
        constraint['type'] = 'eq'
        constraint['fun'] = lambda x: np.linalg.norm(x[nb_jnts + self.NB_CONTACTS * 2 + self.ND_DIM_3D:]) - 1
        return constraint

    def _global_collision_constraint(self, object: trimesh.Trimesh, active_joints: tp.List[int]) -> tp.Dict:

        def constrain_cb(x: tp.List[float], object: trimesh.Trimesh, active_joints: tp.List[int]) -> float:
            nb_joints = len(active_joints)
            quat = x[nb_joints + self.NB_CONTACTS * 2 + self.ND_DIM_3D:]
            trans = x[nb_joints + self.NB_CONTACTS * 2: nb_joints + self.NB_CONTACTS * 2 + self.ND_DIM_3D]
            obj_transform = GraspSynthesizer.trans_quat_to_mat(trans, quat)
            obj_in_place = object.copy()
            obj_in_place.apply_transform(obj_transform)
            q = self._robot_model.qz()
            q[active_joints] = x[:nb_joints]
            return self._robot_model.check_collisions(q, obj_in_place)

        constraint = {}
        constraint['type'] = 'ineq'
        constraint['fun'] = lambda x: constrain_cb(x, object, active_joints)

        return constraint

    def _self_collision_constraint(self, active_joints: tp.List[int]) -> tp.Dict:

        def constrain_cb(x: tp.List[float], active_joints: tp.List[int]) -> float:
            nb_joints = len(active_joints)
            q = self._robot_model.qz()
            q[active_joints] = x[:nb_joints]
            return self._robot_model.check_self_collisions(q)

        constraint = {}
        constraint['type'] = 'ineq'
        constraint['fun'] = lambda x: constrain_cb(x, active_joints)
        return constraint

    def _robot_object_collision_constraint(self, object: trimesh.Trimesh, active_joints: tp.List[int]) -> tp.Dict:

        def constrain_cb(x: tp.List[float], object: trimesh.Trimesh, active_joints: tp.List[int]) -> float:
            nb_joints = len(active_joints)
            quat = x[nb_joints + self.NB_CONTACTS * 2 + self.ND_DIM_3D:]
            trans = x[nb_joints + self.NB_CONTACTS * 2: nb_joints + self.NB_CONTACTS * 2 + self.ND_DIM_3D]
            obj_transform = GraspSynthesizer.trans_quat_to_mat(trans, quat)
            obj_in_place = object.copy()
            obj_in_place.apply_transform(obj_transform)
            q = self._robot_model.qz()
            q[active_joints] = x[:nb_joints]
            return self._robot_model.check_object_collisions(q, obj_in_place)

        constraint = {}
        constraint['type'] = 'ineq'
        constraint['fun'] = lambda x: constrain_cb(x, object, active_joints)

        return constraint

    def _contact_constraint_on_learned_geom(self,
                                            object: trimesh.Trimesh,
                                            active_joints: tp.List[int],
                                            contact_number: int,
                                            link_name: str) -> tp.Dict:

        def constraint_cb(x: tp.List[float],
                          object: trimesh.Trimesh,
                          active_joints: tp.List[int],
                          contact_number: int,
                          link_name: str) -> float:

            nb_joints = len(active_joints)

            # Get contact point on the robot
            contact_param = x[nb_joints+2*contact_number:nb_joints+2*contact_number+2]
            q = self._robot_model.qz()
            q[active_joints] = x[:nb_joints]
            robot_contact_point = self._robot_model.link_param_to_abs(link_name, contact_param[0], contact_param[1], q)

            # Get object at its position
            quat = x[nb_joints + self.NB_CONTACTS * 2 + self.ND_DIM_3D:]
            trans = x[nb_joints + self.NB_CONTACTS * 2: nb_joints + self.NB_CONTACTS * 2 + self.ND_DIM_3D]
            obj_transform = GraspSynthesizer.trans_quat_to_mat(trans, quat)
            obj_in_place = object.copy()
            obj_in_place.apply_transform(obj_transform)

            # Compute distance between object and contact point
            contact_dst = -trimesh.proximity.signed_distance(obj_in_place, robot_contact_point.transpose())[0]

            # Apply some tolerance
            if -self.CONTACT_DST_TOL < contact_dst < self.CONTACT_DST_TOL:
                contact_dst = 0

            return contact_dst

        constraint = {}
        constraint['type'] = 'eq'
        constraint['fun'] = lambda x: constraint_cb(x, object, active_joints, contact_number, link_name)

        return constraint

    def _contact_constraint_on_real_geom(self,
                                         object: trimesh.Trimesh,
                                         active_joints: tp.List[int],
                                         contact_number: int,
                                         link_name: str) -> tp.Dict:

        def constraint_cb(x: tp.List[float],
                          object: trimesh.Trimesh,
                          active_joints: tp.List[int],
                          contact_number: int,
                          link_name: str) -> float:

            nb_joints = len(active_joints)

            # Get contact point on the robot
            contact_param = x[nb_joints+2*contact_number:nb_joints+2*contact_number+2]
            q = self._robot_model.qz()
            q[active_joints] = x[:nb_joints]
            robot_contact_point = self._robot_model.link_param_to_abs(link_name, contact_param[0], contact_param[1], q)
            link_mesh = self._robot_model.get_mesh_at_q(q, link_name)
            real_robot_contact_point, _, _ = trimesh.proximity.closest_point(link_mesh, robot_contact_point.transpose())

            # Get object at its position
            quat = x[nb_joints + self.NB_CONTACTS * 2 + self.ND_DIM_3D:]
            trans = x[nb_joints + self.NB_CONTACTS * 2: nb_joints + self.NB_CONTACTS * 2 + self.ND_DIM_3D]
            obj_transform = GraspSynthesizer.trans_quat_to_mat(trans, quat)
            obj_in_place = object.copy()
            obj_in_place.apply_transform(obj_transform)

            # Compute distance between object and contact point
            contact_dst = -trimesh.proximity.signed_distance(obj_in_place, real_robot_contact_point)[0]

            # Apply some tolerance
            if -self.CONTACT_DST_TOL < contact_dst < self.CONTACT_DST_TOL:
                contact_dst = 0

            return contact_dst

        constraint = {}
        constraint['type'] = 'eq'
        constraint['fun'] = lambda x: constraint_cb(x, object, active_joints, contact_number, link_name)

        return constraint