

## Instructions
- This repo is based on [MuJoCo 2.3.6](https://github.com/deepmind/mujoco)
- Create a new Conda env with Python>= 3.8
- Install mujoco by `pip installl mujoco`
- Absolute path is used here. You will need to change the path for meshes

## Visulization
python -m mujoco.viewer --mjcf=six_finger_hand.xml
roslaunch urdf_tutorial display.launch model:=URDF_finger_llll.urdf



## Notes
compile urdf to .xml for mujoco
https://github.com/wangcongrobot/dual_ur5_husky_mujoco

`./compile ~/research/lasa/mujoco_new/crawling_robot/single_finger/shortest/URDF_finger_ssss_tags.urdf  ~/research/lasa/mujoco_new/crawling_robot/single_finger/shortest/URDF_finger_ssss_tags.xml
`

 - Under folder `singer_finger/`, `URDF_finger_xxxx.urdf `is the urdf file for a single finger.
 - For six fingers, see `descriptions/six_fingers_llll.xml`, which makes 6 copies of the single finger.