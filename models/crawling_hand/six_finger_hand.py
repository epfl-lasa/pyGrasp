import time

from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
from controller import  Robot


model = load_model_from_path("descriptions/five_capsule_finger_hand.xml")
sim = MjSim(model)

viewer = MjViewer(sim)


sim_state = sim.get_state()

while True:
    sim.set_state(sim_state)

    sim.step()
    viewer.render()
    time.sleep(0.002)