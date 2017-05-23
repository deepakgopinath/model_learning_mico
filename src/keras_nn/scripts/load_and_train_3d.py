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

def load_quat_data(datasize, isall):
	num_samples_all = 0
	X_all = np.empty((6,0))
	Y_all = np.empty((6,0))
	path = '/home/deepak/Desktop/Code/model_learning_mico/src/keras_nn/scripts/Data'
	# path =  '/home/deepak/Desktop/NorthwesternStuff/Courses/ArchiveQuarters/Fall2016/DeepLearningUdacity/nn_mico/src/keras_nn/scripts/QuatData'

	
	for filename in glob.glob(os.path.join(path, '*.pkl')):
		with open(filename, 'rb') as openfile:
			data = pickle.load(openfile)
			data = data.T
			X_all = np.concatenate((X_all, data[:,0:-1]), axis=1)
			Y_all = np.concatenate((Y_all, data[:,1:]), axis=1)
			# embed()
			num_samples_all += data.shape[1] - 1

	# embed()
	
	# XTr, XTe, YTrain, YTest = train_test_split(X_all.T, Y_all.T, test_size = 0.0)

	#pick "datasize" number of random points
	# embed()
	random.seed(23)
	if bool(isall):
		datasize = X_all.shape[1]

	embed()

	idxs = np.array(random.sample(range(X_all.shape[1]), datasize))
	X = X_all[:,idxs].T
	Y = Y_all[:,idxs].T

  	# XTr, XTe, YTr, YTe = train_test_split(X.T, Y.T, test_size = 0.0)
  	# embed()	
	XTr = X[:,0:3] #for branch one
	UTr = X[:,3:6] #for branch two
	Y = Y[:,0:3] #output

	# XTest = XTe[:,0:3]
	# UTest = XTe[:, 3:6]
	# YTest = YTe[:, 0:3]
  	# embed()

  	X = XTr
  	U = UTr
  	

  	# embed()
  	lm = LinearModel(X.shape[1], U.shape[1], Y.shape[1])
  	lm.train([X, U], Y)
 #  	pred_norm = np.zeros(XTest.shape[0])	
	# for i in range(0, XTest.shape[0]):
	# 	x = XTest[i, :].reshape(1, 4)
	# 	u = UTest[i, :].reshape(1, 4)
	# 	pred_norm[i] = np.linalg.norm(lm.predict([x,u]))
	lm.test([X, U], Y)	
	
		
if __name__ == '__main__':
	args = sys.argv
	load_quat_data(int(args[1]), int(args[2]))