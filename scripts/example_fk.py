"""An example on how to use simple FK form the RobotModel in this package
"""
import random

import pyGrasp.utils as pgu
from pyGrasp.robot_model import RobotModel

# Choose your example robot here
SELECTED_ROBOT = pgu.CH_FINGER_LONG_URDF_PATH


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

    # Select random joints and angles for FK
    q_fk = robot_model.random_q()
    link_origin_id = random.randint(0, robot_model.nlinks-2)
    link_goal_id = random.randint(link_origin_id + 1, robot_model.nlinks-1)
    link_origin = robot_model.links[link_origin_id]
    link_goal = robot_model.links[link_goal_id]

    # Perform FK
    fk_result = robot_model.fkine(q_fk, link_goal, link_origin)
    fk_jacob = robot_model.jacobe(q_fk, link_goal, link_origin)

    # Display results
    print(f"Result for forward kinematic\n"
          f"Origin link: {robot_model.links[link_origin_id].name}\n"
          f"Goal  link: {robot_model.links[link_goal_id].name}\n"
          f"Joint position: {q_fk}\n\n")
    print("SE3 transform matrix:\n", fk_result)
    print("Geometrical jacobian: \n", fk_jacob)

    # Plot FK
    robot_model.plot(q_fk, backend='swift', block=True)


if __name__ == "__main__":
    main()
