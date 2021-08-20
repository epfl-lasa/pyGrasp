import numpy as np
from numpy import linalg as LA

# Calculate sin, cos, tan in degree
def sind(d):
	# d: angle represented in degree
	return np.sin(d*np.pi/180.)

def cosd(d):
	return np.cos(d*np.pi/180.)

def tand(d):
	return np.tan(d*np.pi/180.)

def arcsind(x):
	# x is the value
	return np.degrees(np.arcsin(x))

def arccosd(x):
	return np.degrees(np.arccos(x))

def arctand(x):
	return np.degrees(np.arctan(x))

'''
Rotation Matrix
'''
# In radius
def rotxr(r):
	Rx = [[1., 0., 0.], [0., np.cos(r), -np.sin(r)], [0., np.sin(r), np.cos(r)]]
	return Rx

def rotyr(r):
	Ry = [[np.cos(r), 0., np.sin(r)], [0., 1., 0.], [-np.sin(r), 0., np.cos(r)]]
	return Ry

def rotzr(r):
	# Calculate the rotation matrix of rotating 'r' w.r.t. the Z-axis, r in radius
	Rz = [[np.cos(r), -np.sin(r), 0.], [np.sin(r), np.cos(r), 0.], [0., 0., 1.]]
	return Rz
	
# In degree
def rotxd(d):
	Rx = [[1., 0., 0.], [0., cosd(d), -sind(d)], [0., sind(d), cosd(d)]]
	return Rx

def rotyd(d):
	Ry = [[cosd(d), 0., sind(d)], [0., 1., 0.], [-sind(d), 0., cosd(d)]]
	return Ry

def rotzd(d):
	# Calculate the rotation matrix of rotating 'r' w.r.t. the Z-axis, r in radius
	Rz = [[cosd(d), -sind(d), 0.], [sind(d), cosd(d), 0.], [0., 0., 1.]]
	return Rz