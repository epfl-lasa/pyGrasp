import time
import mujoco
from mujoco import viewer

# from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
from controller import  Robot

xml_path = 'descriptions/six_finger_hand_llll.xml'
# xml_path = "descriptions/five_capsule_finger_hand.xml"
#xml_path = 'single_finger/longest/URDF_finger_llll_tags.urdf'
#xml_path = 'single_finger/longest/URDF_finger_llll.urdf'
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
mujoco.mj_step(model, data)

# viewer.launch(model, data)
view = viewer.launch_passive(model, data)



while True:
    mujoco.mj_step(model, data)
    view.sync()
    time.sleep(0.002)