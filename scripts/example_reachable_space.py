"""An example on how to compute the reachable space of a robot with this package.
"""
import random

import pyGrasp.utils as pgu
from pyGrasp.robot_model import RobotModel
from pyGrasp.reachable_spaces import ReachableSpace


# Choose your example robot here
SELECTED_ROBOT = pgu.IIWA7_URDF_PATH  # Find all possible robot in the utils.py file


def main() -> None:
    """Gets the model of a robot, print it and computes the reachable space for every link.
    Reachable space are the shown sequentially in different windows.
    You need to close the previous window to see the next one.
    """

    random.seed(0)  # For repeatability

    # Load urdf
    if SELECTED_ROBOT.folder.is_dir() and SELECTED_ROBOT.file_path.is_file():
        robot_model = RobotModel(SELECTED_ROBOT.folder, SELECTED_ROBOT.file_path)
        print("Loaded robot model:")
        print(robot_model)
    else:
        raise FileNotFoundError(f"URDF provided is not a valid file path: {SELECTED_ROBOT}")

    # Create reachable space
    rs = ReachableSpace(robot_model)
    rs.compute_rs(force_recompute=True)

    # Show all rs to check
    rs.show_all_rs()


if __name__ == "__main__":
    main()