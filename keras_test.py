# from keras.models import Sequential
#
# model = Sequential()
#
#
# from keras.layers import Dense, Activation
#
#
# model.add(Dense(units=64, input_dim=100))
# model.add(Activation('relu'))
# model.add(Dense(units=10))
# model.add(  Activation('softmax'))
#
# model.compile(loss='category_crossentrophy', optimizer='sgd', metrics=['accuracy'])
#

'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.preprocessing import image
import scipy.misc

from keras.datasets import mnist
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from keras import backend as K


import numpy as np

batch_size = 128
num_classes = 10
epochs = 15

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Model from samples
# model = Sequential()
#
#
# model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))

# # Model from net (https://www.kaggle.com/aharless/keras-mnist-cnn-from-peter-grenholm)
# model = Sequential()
#
# model.add(Conv2D(filters = 32, kernel_size = (3, 3), input_shape = (28, 28, 1), padding='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(filters = 32, kernel_size = (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))
#
# model.add(Conv2D(filters = 64, kernel_size = (3, 3)))
# model.add(Activation('relu'))
# model.add(Conv2D(filters = 64, kernel_size = (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))
#
# model.add(Flatten())
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10))
# model.add(Activation('softmax'))
#
#
#
# model.compile(loss=keras.losses.categorical_crossentropy,
#               # optimizer=keras.optimizers.Adadelta(),
#               # optimizer=keras.optimizers.Adam(), # OR
#               optimizer=keras.optimizers.Adagrad(), # OR
#               # optimizer=keras.optimizers.SGD(), # OR
#               metrics=['accuracy'])
#
# tensorboard = keras.callbacks.TensorBoard(batch_size=batch_size)
# checkpoints = keras.callbacks.ModelCheckpoint('checkpoint_adagrad.h5', save_best_only=True)
#
# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           callbacks=[tensorboard, checkpoints],
#           validation_data=(x_test, y_test)
#           )

# model.save('my_model_adagrad.h5')
# del model
#

# model = load_model('my_model_adadelta.h5')
# score = model.evaluate(x_test, y_test, verbose=0)
#
# print('Adadelta test loss:', score[0])
# print('Adadelta test accuracy:', score[1])


# Read images to be tested
# image = image.load_img('test_images/3_1.png', target_size=(28, 28))
# image2 = np.array(image)
# image2  = image2 / 255

image3 = scipy.misc.imread('test_images/9_4.png')
image3 = image3 / 255


model = load_model('my_model_adadelta.h5')
# classes = model.predict(x_test, batch_size=128)
# classes = model.predict_classes(image2.reshape(1, 28, 28, 1))
classes2 = model.predict_classes(image3.reshape(1, 28, 28, 1))



# print(classes)
print(classes2)
#
# # get all the predictions (indexes of biggest number [0...9] = predicted number)
# predictions = np.argmax(classes, axis=1)
# real = y_test = np.argmax(y_test, axis=1) # real classes
# diff = real - predictions # if not zero, then the prediction was wrong
#
# for i, value in enumerate(diff):
#     if value != 0:
#         print(str(i))

print('Finished')





