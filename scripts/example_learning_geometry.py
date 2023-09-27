"""This file contains an example of a robot learning the geometry of its links through gpr"""
import random

import pyGrasp.utils as pgu
from pyGrasp.robot_model import RobotModel

# Choose your example robot here
SELECTED_ROBOT = pgu.CH_FINGER_LONG_URDF_PATH # Find all possible robot in the utils.py file


def main() -> None:
    """Gets the model of a robot, print it, learns the geometry of the links and then display them.
    Display is done in a separated window for each link
    """

    random.seed(0)  # For repeatability

    # Load urdf
    if SELECTED_ROBOT.folder.is_dir() and SELECTED_ROBOT.file_path.is_file():
        robot_model = RobotModel(SELECTED_ROBOT.folder, SELECTED_ROBOT.file_path)
        print("Loaded robot model:")
        print(robot_model)
    else:
        raise FileNotFoundError(f"URDF provided is not a valid file path: {SELECTED_ROBOT}")

    # Learning robot geometry and plot
    robot_model.learn_geometry(nb_learning_pts=1000, verbose=True, force_recompute=False)
    robot_model.show_geometries()


if __name__ == "__main__":
    main()
