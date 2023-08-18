# This script creates an allegro hand model.
import sys
import time
import numpy as np
from numpy import linalg as LA
from numerical_computation import *

class allegro_hand(object):
	def __init__(self, T):
		# "mySGallegroLeft" 
		# define all hand parameters
		self.T = T # transformation of base
		self.hand_type = 'AllegroHandLeft'
		self.FinTipDim = 28 # fingertip diameter
		self.phalanx_radius = self.FinTipDim

		self.cc = {'rho': self.FinTipDim/2., 'phi': 0., 'alp': 0.5} # cylindrical coordinates

		self.DHpars = [None]*4
		self.base = [None]*4
		self.Fingers = [None]*4 # This contains all finger objects

		'''
		Fingers 0: Thumb
		'''
		x11 = 16.958; y11 = -73.288; z11 = 18.2
		x12 = 72.147; y12 = -78.116; z12 = 13.2
		x13 = 72.147; y13 = -78.116; z13 = 13.2
		x14 =123.351; y14 = -82.596; z14 = 13.2
		
		rotthumb = -5.*np.pi/180. # in radius

		d11 = -np.sqrt((z11-z12)**2+(y11-y12)**2)
		a11 = z11-z12
		x12dh = x11 - d11*np.sin(rotthumb)
		y12dh = y11 + d11*np.cos(rotthumb)
		a12 = -np.sqrt((x12-x11)**2 + (z12-z11)**2)
		a13 = np.sqrt((x14-x13)**2 + (y14-y13)**2)
		a14 = 59.3

		self.base[0] = np.array([[1., 0., 0., x11], [0., 1., 0., y11], [0., 0., 1., z11], [0., 0., 0., 1.]]) # x11: base[0][0][3]
		self.base[0][0:3,0:3] = rotzr(rotthumb).dot(rotzr(-np.pi/2.)).dot(rotyr(-np.pi/2.))
		self.DHpars[0] = np.array([[np.pi/2, a11, 0., d11], [np.pi/2, 0., np.pi/2, a12], [0., a13, -np.pi/2, 0.], [0., a14, 0., 0.]]) # alpha, a, theta, d

		'''
		Fingers 1: Index finger
		'''
		x21 = 45.098; y21 = 14.293
		a22 = 54.0; a23 = 38.4; a24 = 43.7
		rotbaseindex = -5.*np.pi/180.

		self.base[1] = np.array([[1., 0., 0., x21], [0., 1., 0., y21], [0., 0., 1., 0.], [0., 0., 0., 1.]])
		self.base[1][0:3,0:3] = rotzr(rotbaseindex).dot(rotzr(-np.pi/2.)).dot(rotyr(-np.pi/2.))
		self.DHpars[1] = np.array([[np.pi/2., 0., 0., 0.], [0., a22, np.pi/2., 0.], [0., a23, 0., 0.], [0., a24, 0., 0.]])

		'''
		Fingers 2: Middle Finger
		'''
		x31 = 0.; y31 = 16.6
		a32 = a22; a33 = a23; a34 = a24

		self.base[2] = np.array([[1., 0., 0., x31], [0., 1., 0., y31], [0., 0., 1., 0.], [0., 0., 0., 1.]])
		self.base[2][0:3,0:3] = rotzr(-np.pi/2.).dot(rotyr(-np.pi/2.))
		self.DHpars[2] = np.array([[np.pi/2., 0., 0., 0.], [0., a32, np.pi/2., 0.], [0., a33, 0., 0.], [0., a34, 0., 0.]])

		'''
		Fingers 3: Ring Finger
		'''
		x41 = -45.098; y41 = 14.293
		a42 = a22; a43 = a23; a44 = a24
		rotbaselast = 5.*pi/180.

		self.base[3] = np.array([[1., 0., 0., x41], [0., 1., 0., y41], [0., 0., 1., 0.], [0., 0., 0., 1.]])
		self.base[3][0:3,0:3] = rotzr(rotbaselast).dot(rotzr(-np.pi/2.)).dot(rotyr(-np.pi/2.))
		self.DHpars[3] = np.array([[np.pi/2., 0., 0., 0.], [0., a42, np.pi/2., 0.], [0., a43, 0., 0.], [0., a44, 0., 0.]])

	def construct_hand_model(self):
		# Construct hand and fingers
		for idx in range(len(self.DHpars)):
			if idx == 0: # thumb
				lb = np.array([0.3635738998060688, -0.20504289759570773, -0.28972295140796106, -0.26220637207693537])
				ub = np.array([1.4968131524486665, 1.2630997544532125, 1.7440185506322363, 1.8199110516903878])
			else:
				lb = np.array([-0.59471316618668479, -0.29691276729768068, -0.27401187224153672, -0.32753605719833834])
				ub = np.array([0.57181227113054078, 1.7367399715833842, 1.8098808147084331, 1.71854352396125431])
			q = (lb + ub)/2.
			self.Fingers[idx] = make_finger(self.DHpars[idx], self.T.dot(self.base[idx]), q, idx, lb, ub, '', [], [], self.cc, self.hand_type)
		self.hand = make_hand(self.Fingers, self.T)
		self.hand = make_palm(self.hand)


def make_hand(Fingers, T):
	pass


def make_finger(DHpars, T, q, idx, lb, ub, eval_type, active_joints, contacted_link, cc, hand_type):
	pass

def make_palm(hand):
	pass

def move_hand():
	pass

def construct_reachability_map():
	pass

def construct_self_collision_map():
	pass
