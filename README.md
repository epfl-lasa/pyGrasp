# pyGrasp

This project exploits the kinematic redundancy of the robotic system to enhance its dexterity in task performance, such as grasping single or multiple objects, or multitasking.
It is the Python realization and generalization of [1].

## Project Structure
The project contains the following main functional modules:

### 1. Modeling
- **Robot**: given a robot description file (URDF or xacro), this functional module constructs and stores the full kinematic tree of the robot. For now, three robots are considered:
  - Robotic hand: Allegro hand, human hand
  - Robotic arm: KUKA iiwa, with or w/o the Allegro hand
  - Humanoid robot: iCub
- **Object**: the target object can be represented by (1) a group of geometric primitives (e.g., spheres, cylinders, etc.), and the contact on the object surface can have a closed-form expression; or (2) point cloud data, then a spherical coordinate can be constructed by training a model of the radius $r$, as a function of $theta$ and $phi$.
- **Contact**: for now, *point contact with friction* model is used (torsional friction is optional).

### 2. Analysis of Kinematics
Offline analysis of the robot kinematic model to construct:
- **Reachability map**
- **Opposition space**
- **Self-collision map**

### 3. Grasp Synthesis
Given the robot and the model of the target object, this functional module deals with:
- **Problem formulation**
- **Solving optimization**: currently, `SciPy` package is used to solve the optimization problem with `scipy.optimize()`.

### 4. Sequential Grasp Optimization
- Generates a sequence of grasps to achieve the grasping of multiple objects;
- Optimize overall kinematic efficiency to maximize the number of objects to grasp, or the number of subtasks to execute.

### 5. Collision-free Path Planning
Generate a collision-free path that drives the robotic system from its current configuration to grasp the target object.
Three types of collision are considered:
- **Self collision**: collision among robot links;
- **Robot-object collision**: collision between the target object and any robotic links;
- **Object-object collision**: collision between the target object and any previously grasped objects.

### 6. Motion Generation and Control
Control the robotic system to move along the planned collision-free path, and grasp with compliance and robustness.

## Expected Features
Here list of the features with low priority.
- The coordination of multiple robots
- Manipulation motion generation
- Coordinated multitasking

## Dependencies
1. [hand_dexterity](https://github.com/kunpengyao/hand_dexterity): the MATLAB version code for a multi-fingered dexterous robotic hand to grasp multiple objects
2. [SynGrasp](http://sirslab.dii.unisi.it/syngrasp/): A MATLAB Toolbox for Grasp Analysis of Human and Robotic Hands
3. [CasADi](https://web.casadi.org/): an open-source tool for nonlinear optimization and algorithmic differentiation
4. [scipy.optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html#module-scipy.optimize): provides functions for minimizing (or maximizing) objective functions, possibly subject to constraints. It includes solvers for nonlinear problems (with support for both local and global optimization algorithms), linear programing, constrained and nonlinear least-squares, root finding, and curve fitting.
5. [scikit-learn](https://scikit-learn.org/stable/index.html): for using machine learning tools.

## References
1. Yao, K., & Billard, A. (2023). [Exploiting Kinematic Redundancy for Robotic Grasping of Multiple Objects](https://ieeexplore.ieee.org/abstract/document/10086636). IEEE Transactions on Robotics.
2. Pozzi M, Achilli GM, Valigi MC and Malvezzi M (2022) Modeling and Simulation of Robotic Grasping in Simulink Through Simscape Multibody. Front. Robot. AI 9:873558. doi: 10.3389/frobt.2022.873558

## Get Started

### Requirements
- Python 3.9 or higher

### Activate or create the virtual environment

To setup the virtual environment, here is how it goes:
```bash
python3.9 -m venv .venv          # Create venv (use any python version you like >=3.9)
source .venv/bin/activate        # Activate venv
pip install -r requirements.txt  # Install package requirements
python -m build                  # Build package
pip install -e .                 # Install package. -e for editable, developer mode.
```

Or, for a faster onliner, you can use the following script

```bash
source setup_env.bash
```

### Examples to test modules

Once the environment is setup, you can try running the following scripts:

```bash
python scripts/example_fk.py                 # Run an example of simple forward kinematics
python scripts/example_learning_geometry.py  # Run an example of a robot learning its links geometry and displays results
python scripts/example_extended_fk.py        # Run an example of the forward kinematics to an arbitrary point on the robot
```


### Developpement, issues and bugs

#### Issues and improvements

- [ ] RS more precise through heavier computations
- [x] Fix issue with os not propagating correctly
- [ ] Find an efficient way to get GUI feedback from ssh connection with pyglet
- [x] RS seems unlikely with allegro... pinpoint why (Done, fixed bug regarding fixed links in rs)
  - link_11.0_tip
  - link_7.0_tip
  - link_3.0_tip
  - link_15.0_tip
- [ ] Add iCub geometry
- [ ] Extend framework for as many links as we want
- [ ] Handle redundant code in geometry propagation
- [ ] Figure out how to integrate contact directionnality in OS's
- [ ] Check about abnormal termination of the optimizer
- [x] Add jacobians for quaternion
- [ ] Use self collision map to speed up collision computations
- [ ] Train ML model to compute force closure property
- [ ] Assess computational complexity of just validating force colure after sampling grasp configurations
- [ ] Clean up redundant code in robot model collision checks
- [ ] Differentiate dev from prod env
- [ ] Add robot composition
- [ ] If we represent the orientation as an axis angle, with the axis in sphe we can remove 1 variable and 1 constraint
- [ ] If we represent the object using GP, adding 2 variables, we have a mathematical gradient for the contact constraints
- [ ] Have uniform functions to access different parameters from the list
- [ ] Have a single function to access the object in its correct place
- [ ] Assess to see if we need to have a threshold
- [ ] Adaptive contact tolerance on the constraint

#### TODO list

- [ ] Integrate nn distance function
- [ ] Entire framework as feasability problem
- [ ] Extend the number of contacts (realise for 3-4 contact points)
- [ ] Force closure for multiple contacts
- [ ] Add ICub geometry
