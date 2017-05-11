#!/usr/bin/env python

import numpy as np
import pickle
import glob
from IPython import embed
from sklearn.cross_validation import train_test_split
import os
from LinearModel import LinearModel
import tf.transformations as tfs
npa = np.array

def load_quat_data():
	num_samples_all = 0

	XA_all = np.empty((4,0))
	wA_all = np.empty((6,0)) #omega matrix
	UA_all = np.empty((4,0)) #qrate matrix. 

	e_all = np.empty((4,0))
	u_all = np.empty((4,0))


	X_all = np.empty((8,0))
	Y_all = np.empty((4,0))

	qt_all = np.empty((4,0))
	qtnext_all = np.empty((4,0))
	w_all = np.empty((6,0))
	path = '/home/deepak/Desktop/Code/model_learning_mico/src/keras_nn/scripts/PoseData'
	# path =  '/home/deepak/Desktop/NorthwesternStuff/Courses/ArchiveQuarters/Fall2016/DeepLearningUdacity/nn_mico/src/keras_nn/scripts/QuatData'

	qd = npa([-0.417, 0.467, -0.559,  -0.543])
	HR_qd = compute_HR(qd) #np.matrix
	

	for filename in glob.glob(os.path.join(path, '*.pkl')):
		with open(filename, 'rb') as openfile:
			data_dict = pickle.load(openfile)
			quats = data_dict['orientation'].T
			w = data_dict['input_vel'].T 
			qt_all = np.concatenate((qt_all, quats[:,0:-1]), axis=1)
			w_all = np.concatenate((w_all, w[:,0:-1]), axis=1)
			qtnext_all = np.concatenate((qtnext_all, quats[:,1:]), axis=1)
			# embed()
			num_samples_all += w.shape[1] - 1


	for i in range(0, num_samples_all):
		w = w_all[3:6, i]
		w_mat = np.matrix([ [0, w[2], -w[1], w[0]],
							[-w[2], 0, w[0], w[1]],
							[w[1], -w[0], 0, w[2]],
							[-w[0], -w[1], -w[2], 0]])
		q = qt_all[:,i]
		q = q.reshape(q.size, 1)
		qnext = qtnext_all[:,i]
		qnext = qnext.reshape(qnext.size, 1)

		qdot = 0.5*w_mat*q
		edot = -HR_qd*qdot
		u_all = np.concatenate((u_all, edot), axis=1)
		
		e = npa([0,0,0,1]).reshape(4,1) - tfs.quaternion_multiply(tfs.quaternion_conjugate(q), qd)
		e_all = np.concatenate((e_all, e), axis=1)

		enext = npa([0,0,0,1]).reshape(4,1) - tfs.quaternion_multiply(tfs.quaternion_conjugate(qnext), qd)
		Y_all = np.concatenate((Y_all, enext), axis=1)

		# embed()

	X_all = np.concatenate((e_all, u_all), axis=0)
	embed()


	# X_all = np.concatenate((XA_all, UA_all), axis=0) #stack X and U on top of each other. 8, N size.
	XTr, XTe, YTrain, YTest = train_test_split(X_all.T, Y_all.T, test_size = 0.1)
	# embed()
	XTr = XTr.T 
	XTrain = XTr[0:4, :].T
	UTrain = XTr[4:8, :].T
	XTe = XTe.T
	XTest = XTe[0:4, :].T
	UTest = XTe[4:8, :].T
	
	embed()
	#instantiate model
	lm = LinearModel(XTrain.shape[1], UTrain.shape[1], YTrain.shape[1])
	# embed()
	lm.train([XTrain, UTrain], YTrain)
	# embed()
	pred_norm = np.zeros(XTest.shape[0])	
	for i in range(0, XTest.shape[0]):
		x = XTest[i, :].reshape(1, 4)
		u = UTest[i, :].reshape(1, 4)
		pred_norm[i] = np.linalg.norm(lm.predict([x,u]))

	# embed()
	lm.test([XTest, UTest], YTest)

def compute_HR(q):
	qw = q[3]
	qx = q[0]
	qy = q[1]
	qz = q[2]

	I = np.matrix(np.eye(4))
	HR = qw*I + np.matrix([ [0, -qx, -qy, -qz],
							[qx, 0, -qz, qy],
							[qy, qz, 0, -qx],
							[qz, -qy, qx, 0]])
	return HR

	
		
if __name__ == '__main__':
	load_quat_data()