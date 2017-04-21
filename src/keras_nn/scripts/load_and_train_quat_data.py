#!/usr/bin/env python

import numpy as np
import pickle
import glob
from IPython import embed
from sklearn.cross_validation import train_test_split
import os
from LinearModel import LinearModel

def load_quat_data():
	num_samples_all = 0

	XA_all = np.empty((4,0))
	wA_all = np.empty((3,0)) #omega matrix
	UA_all = np.empty((4,0)) #qrate matrix. 
	X_all = np.empty((8,0))
	Y_all = np.empty((4,0))
	path = '/home/deepak/Desktop/Code/model_learning_mico/src/keras_nn/scripts/QuatData'

	for filename in glob.glob(os.path.join(path, '*.pkl')):
		with open(filename, 'rb') as openfile:
			data_dict = pickle.load(openfile)
			quats = data_dict['orientation']
			w = data_dict['orientation_vel']
			# zero_rows = np.where(~w.any(axis=1))[0]
			# w = np.delete(w, zero_rows, axis=0)
			# quats = np.delete(quats, zero_rows, axis=0)
			w = w.T
			quats = quats.T
			XA_all = np.concatenate((XA_all, quats[:,0:-1]), axis=1)
			wA_all = np.concatenate((wA_all, w[:,0:-1]), axis=1)
			Y_all = np.concatenate((Y_all, quats[:,1:]), axis=1) #samples along column
			num_samples_all += w.shape[1] - 1

	# embed()
	for i in range(0, num_samples_all):
		w = wA_all[:,i]
		w_mat = np.matrix([[0, -w[0], -w[1], -w[2]],
						   [w[0], 0, w[2], -w[1]],
						   [w[1], -w[2], 0, w[0]],
						   [w[2], w[1], -w[0], 0]])
		q = XA_all[:,i]
		q = q.reshape(q.size, 1) #make it proper column vector for matrix multiply
		qdot = 0.5*w_mat*q 
		qdot = np.asarray(qdot)
		# embed()
		UA_all = np.concatenate((UA_all, qdot), axis=1)
		# embed()


	X_all = np.concatenate((XA_all, UA_all), axis=0) #stack X and U on top of each other. 8, N size.
	XTr, XTe, YTrain, YTest = train_test_split(X_all.T, Y_all.T, test_size = 0.1)
	# embed()
	XTr = XTr.T 
	XTrain = XTr[0:4, :].T
	UTrain = XTr[4:8, :].T
	XTe = XTe.T
	XTest = XTe[0:4, :].T
	UTest = XTe[4:8, :].T
	
	# embed()
	#instantiate model
	lm = LinearModel(XTrain.shape[1], UTrain.shape[1], YTrain.shape[1])
	# embed()
	lm.train([XTrain, UTrain], YTrain)
	embed()
	lm.test([XTest, UTest], YTest)
	# embed()
	
		
if __name__ == '__main__':
	load_quat_data()