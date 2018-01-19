# Original project code can be found at: https://github.com/DJTobias/Cherry-Autonomous-Racecar/
# Pretrained Tensorflow models are avialable at: https://drive.google.com/drive/folders/0Bx1XHaRXk3kSY1pfaW9hOVU4RWc


import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard, ModelCheckpoint
# from keras.preprocessing.image import ImageDataGenerator
from load_data import cherryCar_loadData_full


x_train, y_train, _, _ = cherryCar_loadData_full(13987)  # change in API - must pass max number of images, 13987 is full original datasste

x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
x_train /= 255

# image data augmentation
# datagen = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2)
# datagen.fit(x_train)


batch_size = 32
num_epochs = 200


model = Sequential()
model.add(Conv2D(32, (5, 5), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
# model.add(Activation('relu', max_value = 1.0 ))

# opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
opt = keras.optimizers.adam(lr=0.0001)
model.compile(loss='mean_squared_error',
              optimizer=opt)


model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=num_epochs,
          validation_split=0.1,
          shuffle=True,
          callbacks=[TensorBoard(log_dir='tmp/CherryCar'), ModelCheckpoint('model/checkpoint_CherryCar.h5', save_best_only=True)])

# only when using image augmentation
# model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
#                                 steps_per_epoch=len(x_train)/batch_size,
#                                 epochs=num_epochs,
#                                 callbacks=[TensorBoard(log_dir='tmp/CherryCar200'), ModelCheckpoint('model/checkpoint_CherryCar_2.h5', save_best_only=True)])

model.save('model/model_CherryCar.h5')





