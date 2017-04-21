#!/usr/bin/env python

import numpy as np
import pickle
import glob
import os

def load_data():

    num_samples_all = 0
    num_samples_ten = 0 # this is going to be ten, duhhhhhhh

    X_all = np.empty((6,0))
    Y_all = np.empty((6,0))
    X_ten = np.empty((6,0))
    Y_ten = np.empty((6,0))
    path = '/home/deepak/Desktop/NorthwesternStuff/Courses/ArchiveQuarters/Fall2016/DeepLearningUdacity/nn_mico/src/keras_nn/scripts/Data'

    for filename in glob.glob(os.path.join(path, '*.pkl')):
        if 'K.pkl' not in filename and 'A.pkl' not in filename and 'B.pkl' not in filename:
            with open(filename, 'rb') as openfile: 
                data = pickle.load(openfile)
                data = data.T

                X_all = np.concatenate((X_all, data[:,0:-1]), axis=1)
                Y_all = np.concatenate((Y_all, data[:,1:]), axis=1)
                num_samples_all += data.shape[1] - 1

                # if not ('11' in filename or '12' in filename or '13' in filename):
                #     X_ten = np.concatenate((X_ten, data[:,0:-1]), axis=1)
                #     Y_ten = np.concatenate((Y_ten, data[:,1:]), axis=1)
                #     num_samples_ten += data.shape[1]

    return X_all, Y_all, num_samples_all

if __name__ == '__main__':
    load_data()