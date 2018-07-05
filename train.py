import numpy as np
import h5py

from keras.models import Sequential 
from keras.layers import Dense, Activation, Convolution2D, Flatten,MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
import tensorflow as tf
import os
import matplotlib.pyplot as plt 

trainh5_path = "train_data.h5" 
testh5_path = "test_data.h5"

def read_hdf5(path):
    data = h5py.File(path,'r')
    feature = np.array(data['feature'])
    label = np.array(data['label'])
    feature = feature.reshape(-1,1,64,64)/255.
    lb = np.zeros(shape=(len(label),50))
    j=0
    for i in label:
        lb[j][i-1] = 1
        j+=1
    return feature, lb

tn_f,tn_lb = read_hdf5(trainh5_path)
tt_f,tt_lb = read_hdf5(testh5_path)


model = Sequential()
model.add(Convolution2D(batch_input_shape=(None,1,64,64),filters=64,kernel_size=(3,3),padding='same',dilation_rate=2,data_format='channels_first'))
model.add(Convolution2D(filters=64,kernel_size=(3,3),padding='same',data_format='channels_first'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2),strides=None,padding='same',data_format='channels_first'))

model.add(Convolution2D(filters=128,kernel_size=(3,3),padding='same',data_format='channels_first'))
model.add(Convolution2D(filters=128,kernel_size=(3,3),padding='same',data_format='channels_first'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2),strides=None,padding='same',data_format='channels_first'))

model.add(Convolution2D(filters=256,kernel_size=(3,3),padding='same',data_format='channels_first'))
model.add(Convolution2D(filters=256,kernel_size=(3,3),padding='same',data_format='channels_first'))
model.add(Convolution2D(filters=256,kernel_size=(3,3),padding='same',data_format='channels_first'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2),strides=None,padding='same',data_format='channels_first'))


model.add(Flatten())

model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('softmax'))

model.summary()

adam = Adam(lr=1e-4)

model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

print('Training ......')

model.fit(tn_f,tn_lb,epochs=50,batch_size=50,verbose=1)

print('Testing ......')

loss,acc = model.evaluate(tt_f,tt_lb)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', acc)

model.save('train.h5')
