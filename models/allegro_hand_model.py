# This script creates an allegro hand model.
import sys
import time
import numpy as np
from numpy import linalg as LA


class allegro_hand(object):

	def __init__(self):
		# define all hand parameters

		self.hand_name = 'AllegroHandLeft'
		self.FinTipDim = 28 # fingertip dimension





	def make_hand(self):
		# make hand by constructing fingers and palm
		pass

	def move_hand(self, q):
		# move hand to a position q
		pass

	def construct_self_collision_map(self):
		pass

	def construct_reachability_map(self):
		pass
