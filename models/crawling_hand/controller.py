import numpy as np
import mujoco





class Robot:
    def __init__(self, m: mujoco._structs.MjModel, d: mujoco._structs.MjModel, view, obj_names=[], auto_sync=True,
                 q0=None):
        self.m = m
        self.d = d
        self.view = view
        self.auto_sync = auto_sync

        self.n = len(self.q)
        if q0 is None:
            self.q0 = np.zeros(self.n)

        self.modify_joint(self.q0)  # set the initial joint positions
        self.step()
        self.view.sync()

        # self.viewer_setup()

    def step(self):
        mujoco.mj_step(self.m, self.d)  # run one-step dynamics simulation


    def modify_joint(self, joints: np.ndarray) -> None:
        """
        :param joints: (7,) or (16,) or (23,), modify joints for iiwa or/and allegro hand
        :return:
        """
        assert len(joints) == self.n
        self.d.qpos[7: 7+self.n] = joints

    def send_torque(self, torque):
        """
        control joints by torque and send to mujoco, then run a step.
        input the joint control torque
        Here I imply that the joints with id from 0 to n are the actuators.
        so, please put robot xml before the object xml.
        todo, use joint_name2id to avoid this problem
        :param torque:  (n, ) numpy array
        :return:
        """
        self.d.ctrl[:len(torque)] = torque
        mujoco.mj_step(self.m, self.d)
        if self.auto_sync:
            self.view.sync()


    def joint_impedance_control(self, q, dq=None, k= 0.5):
        kp = np.ones(self.n) * 0.4 * k
        kd = 2 * np.sqrt(kp) * 0.01
        if dq is None:
            dq = np.zeros(self.n)

        error_q = q - self.q
        error_dq = dq - self.dq

        qacc_des = kp * error_q + kd * error_dq + self.C

        self.send_torque(qacc_des)





    @property
    def q(self):
        """
        iiwa joint angles
        :return: (10, ), numpy array
        """
        return self.d.qpos[7:]  # noting that the order of joints is based on the order in *.xml file

    @property
    def dq(self):
        """
        iiwa joint velocities
        :return: (7, )
        """
        return self.d.qvel[6:]

    @property
    def C(self):
        """
        for iiwa, C(qpos,qvel), Coriolis, computed by mj_fwdVelocity/mj_rne (without acceleration)
        :return: (7, )
        """
        return self.d.qfrc_bias[6:]

