"""An example on how to use oppositions spaces in this package
"""
import random

import pyGrasp.utils as pgu
from pyGrasp.robot_model import RobotModel
from pyGrasp.opposition_spaces import OppositionSpace

# Choose your example robot here
SELECTED_ROBOT = pgu.IIWA7_URDF_PATH  # Find all possible robot in the utils.py file


def main() -> None:
    """Gets the model of the IIWA, print it and performs FK to a random link.
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
    opp_s = OppositionSpace(robot_model)
    opp_s.compute_os(force_recompute=True)

    # Get the best opposition space
    link_for_grasp = opp_s.get_best_os(obj_diameter=0.3,
                                       excluded_links=[])  # Names of the links to exclude from the search

    # Link grasp can also be called with a point cloud as follow:
    # link_for_grasp = opp_s.get_best_os(point_cloud=my_point_cloud, excluded_links=[])
    # Point cloud is an array-like of dim (n, 3)

    if link_for_grasp is not None:
        print(f"Links with the best opposition space: {link_for_grasp[0]} {link_for_grasp[1]}")
    else:
        print("No links satisfying the grasp were found")


if __name__ == "__main__":
    main()
