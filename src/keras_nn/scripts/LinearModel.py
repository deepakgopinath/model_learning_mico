#!/usr/bin/env python

import numpy as np
from keras.layers import Dense, Input, Merge
from keras.layers.merge import add
from keras.models import Model, Sequential
from keras.optimizers import SGD
from IPython import embed
# import tf.transformations as tfs
import tensorflow as tf


class LinearModel(object):
	"""docstring for LinearModel"""
	def __init__(self, dimA, dimU, dimY):
		super(LinearModel, self).__init__()
		self.input_A_dim = dimA
		self.input_U_dim = dimU
		self.output_dim = dimY

		self.input_A = Input(name='inputA', shape=(self.input_A_dim,), dtype='float32')
		self.out_A = Dense(self.output_dim, activation='linear', name='outA')(self.input_A)

		self.input_U = Input(name='inputU', shape=(self.input_U_dim,), dtype='float32')
		self.out_U = Dense(self.output_dim, activation='linear', name='outU')(self.input_U)

		self.out = add([self.out_A, self.out_U])
		self.m = Model([self.input_A, self.input_U], self.out)
		# self.m.compile(loss = self.cost_function, optimizer = 'sgd') 
		self.m.compile(loss = 'mse', optimizer = 'sgd')

	

	def train(self, trainX, trainY):
		print "IN TRAINING"
		#might need a different cost function that mse to take into account the SO(3) nature of quaternions
		self.m.fit(trainX, trainY,epochs = 300, batch_size=5)

	def cost_function(self, y_true, y_pred):
		# embed()
		return tf.reduce_mean(tf.square(y_pred - y_true), reduction_indices = 1, keep_dims = False) # + tf.abs((1 - tf.norm(y_pred, ord='euclidean')))
		

	def test(self, testX, testY):
		print "THE LEARNED WEIGHTS ARE", self.m.get_weights()
		print "MSE ON TEST SET IS ", self.m.evaluate(testX, testY, verbose=0)

	def predict(self, X):
		return self.m.predict(X, verbose = 0)



