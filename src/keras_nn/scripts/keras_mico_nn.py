'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

# from __future__ import print_function

# import keras
# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.optimizers import RMSprop
# from IPython import embed

# batch_size = 128
# num_classes = 10
# epochs = 20

# # the data, shuffled and split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# embed()
# x_train = x_train.reshape(60000, 784)
# x_test = x_test.reshape(10000, 784)
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')

# # convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

# model = Sequential()
# model.add(Dense(512, activation='relu', input_shape=(784,)))
# model.add(Dropout(0.2))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='softmax'))

# model.summary()

# model.compile(loss='categorical_crossentropy',
#               optimizer=RMSprop(),
#               metrics=['accuracy'])

# history = model.fit(x_train, y_train,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1,
#                     validation_data=(x_test, y_test))
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Merge, Activation
from keras.layers.merge import add
from keras.optimizers import RMSprop
from IPython import embed
from load_data import load_data
import numpy as np 

batch_size = 128
epochs = 20


X_all, Y_all, num_samples_all= load_data()

shuffle_index = np.random.choice(num_samples_all, num_samples_all, replace=False)

num_train_samples =  int(num_samples_all*0.8)

#shuffle all data points.

X_all = X_all.transpose()
X_all = X_all[shuffle_index, :]
Y_all = Y_all.transpose()
Y_all = Y_all[shuffle_index, :]

#create training and test set
X_train = X_all[0:num_train_samples, :]
Y_train = Y_all[0:num_train_samples, :]
X_test = X_all[num_train_samples:, :]
Y_test = Y_all[num_train_samples:, :]

#split training to two branches
XA_train = X_train[:,0:3]
XB_train = X_train[:,3:6]
Y_train = Y_train[:,0:3]

#splot testing to two branhces
XA_test = X_test[:,0:3]
XB_test = X_test[:, 3:6]
Y_test = Y_test[:, 0:3]

# input_data = Input(name='the_input', shape=(3,), dtype='float32')
# hidden = Dense(3, input_shape=(4,))(input_data)
# output_A = Activation('linear', name='linear')(hidden)
# model_A = Model(inputs=input_data, outputs=output_A)


#A_branch
model_A = Sequential()
model_A.add(Dense(XA_train.shape[1], name='input_A', activation='linear', input_shape=(XA_train.shape[1],))) #input shape is 3 by 1
embed()
# model_A.add(Dense(XA_train.shape[1], name='hidden', activation='linear'))
# embed()
# model_A.add(Dense(Y_train.shape[1], name='output_A', activation='linear'))

# model_B = Sequential()
# model_B.add(Dense(XB_train.shape[1], name='input_B', activation='linear', input_shape=(XB_train.shape[1],)))
# # model_B.add(Dense(XB_train.shape[1], name='hidden', activation='linear'))
# model_B.add(Dense(Y_train.shape[1], name='output_B',  activation='linear'))

# embed()
# model = Sequential()




# embed()