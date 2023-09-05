"""An example on how to use the extended FK from this package
The extended FK is an FK from any base link to any point on a tip link on the robot.
"""
import random
import numpy as np
import matplotlib.pyplot as plt

import pyGrasp.utils as pgu
from pyGrasp.robot_model import RobotModel

# Choose your example robot here
SELECTED_ROBOT = pgu.ALLEGRO_LEFT_URDF_PATH


def main() -> None:
    """Gets the model of the IIWA, print it and performs an extended FK to
    a random point on a random link of the robot.
    """

    random.seed(0)  # For repeatability

    # Load urdf
    if SELECTED_ROBOT.folder.is_dir() and SELECTED_ROBOT.file_path.is_file():
        robot_model = RobotModel(SELECTED_ROBOT.folder, SELECTED_ROBOT.file_path)
        print("Loaded robot model:")
        print(robot_model)
    else:
        raise FileNotFoundError(f"URDF provided is not a valid file path: {SELECTED_ROBOT}")

    # Explicitly ask to learn geometries
    robot_model.learn_geometry(verbose=True)

    # Select random joints and angles for FK
    q_fk = robot_model.random_q()
    link_goal_id = random.randint(1, robot_model.nlinks)
    link_goal = robot_model.links[link_goal_id]
    theta = random.uniform(-np.pi, np.pi)
    phi = random.uniform(0, np.pi)

    # Perform extended fk
    robot_model.extended_fk(q_fk, theta, phi, tip_link=link_goal, plot_result=True)
    plt.show()


if __name__ == "__main__":
    main()
