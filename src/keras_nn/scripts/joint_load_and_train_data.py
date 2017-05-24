#!/usr/bin/env python

import numpy as np
import pickle
import glob
from IPython import embed
from sklearn.cross_validation import train_test_split
import os
from LinearModel import LinearModel
import random
import sys

def load_joint_data():
	num_samples_all = 0

	X_all = np.empty((12,0))
	XA_all = np.empty((6,0))
	UB_all = np.empty((6,0))
 	Y_all = np.empty((6,0))

	# path = '/home/deepak/Desktop/Code/model_learning_mico/src/keras_nn/scripts/JointData'
	path = '/home/deepak/Desktop/Research/RIC/CodeBase/model_learning_mico/src/keras_nn/scripts/JointData'
	for filename in glob.glob(os.path.join(path, '*.pkl')):
		with open(filename, 'rb') as openfile:
			data_dict = pickle.load(openfile)
			jp = data_dict['joint_angles'].T
			jv = data_dict['joint_velocities'].T

			XA_all = np.concatenate((XA_all, jp[:,0:-1]), axis=1)
			Y_all = np.concatenate((Y_all, jp[:,1:]), axis=1)
			if jv.shape[1] > jp.shape[1]:
				UB_all = np.concatenate((UB_all, jv[:,0:-2]), axis=1)
			else:
				UB_all = np.concatenate((UB_all, jv[:,0:-1]), axis=1)
			# embed()
			# embed()

	# embed()
	X_all = np.concatenate((XA_all, UB_all), axis=0)
	
	XTr, XTe, YTrain, YTest = train_test_split(X_all.T, Y_all.T, test_size = 0.1)
	# embed()
	XTr = XTr.T 
	XTrain = XTr[0:6, :].T
	UTrain = XTr[6:12, :].T
	XTe = XTe.T
	XTest = XTe[0:6, :].T
	UTest = XTe[6:12, :].T
	embed()
	lm = LinearModel(XTrain.shape[1], UTrain.shape[1], YTrain.shape[1])
	lm.train([XTrain, UTrain], YTrain)
	lm.test([XTest, UTest], YTest)


if __name__ == '__main__':
	args = sys.argv
	load_joint_data()

        