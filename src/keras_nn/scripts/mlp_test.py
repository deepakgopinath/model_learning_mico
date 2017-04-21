import numpy as np
from keras.layers import Dense, Input, Merge
from keras.layers.merge import add
from keras.models import Model, Sequential
from keras.optimizers import SGD
from IPython import embed
from load_data import load_data
import random

batch_size = 128
epochs = 20

X_all, Y_all, num_samples_all= load_data()

shuffle_index = np.random.choice(num_samples_all, num_samples_all, replace=False) #randomize data to remove correlations between subsequent points

num_train_samples =  int(num_samples_all*1.0) #make trainings and test set. 

#shuffle all data points

X_all = X_all.transpose() # correct input data shape
X_all = X_all[shuffle_index, :] #shuffle the data points
Y_all = Y_all.transpose() #correct output data shape.
Y_all = Y_all[shuffle_index, :]

#extract training and test set

X_train = X_all[0:num_train_samples, :]
Y_train = Y_all[0:num_train_samples, :]
X_test = X_all[num_train_samples:, :]
Y_test = Y_all[num_train_samples:, :]

#split training into two branches
XA_train = X_train[:,0:3] #for branch one
XB_train = X_train[:,3:6] #for branch two
Y_train = Y_train[:,0:3] #output

#split testing into two branches
XA_test = X_test[:,0:3]
XB_test = X_test[:, 3:6]
Y_test = Y_test[:, 0:3]


X = XA_train
U = XB_train
Y = Y_train

# X = np.random.random((3000,2))
# U = np.random.random((3000,3))
# W1 = np.array([[2,1],[3,2]])
# W2 = np.array([[3,2,1],[4,3,2]])
# yX = np.dot(W1, X.transpose()).transpose() 
# yU = np.dot(W2, U.transpose()).transpose()
# Y = yX + yU
# yA = np.dot(xA, [2., 3.])
embed()

# print x[:5]
# lm = Sequential([Dense(2, input_shape=(2,))])
# lm.compile(optimizer=SGD(lr=0.1), loss='mse')
# print lm.evaluate(X, yX, verbose=0)
# lm.fit(X, yX, nb_epoch=1, batch_size=1)
# print lm.get_weights()
# print lm.evaluate(X, yX, verbose=0)

# embed()

# print "A MODEL"

# mA = Model(input_A, inner_A)
# mA.compile(loss = 'mse', optimizer='sgd')
# mA.fit(X,yX, epochs=5, batch_size=1)
# print mA.get_weights()
# print mA.evaluate(X,yX,verbose=0)

# embed()


# print "U MODEL"

# mU = Model(input_U, inner_U)
# mU.compile(loss ='mse', optimizer='sgd')
# mU.fit(U, yU, epochs=5, batch_size=1)
# print mU.get_weights()
# print mU.evaluate(U,yU,verbose=0)

# embed()

print "COMBINED MODEL"
input_A = Input(name='inputA', shape=(3,), dtype='float32')
out_A = Dense(3, activation='linear', name='outA')(input_A)
input_U = Input(name='inputU', shape=(3,), dtype='float32')
out_U = Dense(3, activation='linear', name='outU')(input_U)
out = add([out_A, out_U])
m = Model([input_A, input_U], out)
embed()
m.compile(loss='mse', optimizer='sgd')
m.fit([X, U], Y,epochs = 2000, batch_size=10)
print m.get_weights()
print m.evaluate([XA_test,XB_test], Y_test, verbose=0)
embed()
