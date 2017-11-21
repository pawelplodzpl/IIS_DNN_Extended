from keras import callbacks
from keras.models import load_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, Sequential
from keras import backend as K
from keras.datasets import mnist
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt


(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
# x_train = (x_train.astype('float32') - 127.5) / 127.5
# x_test = (x_test.astype('float32') - 127.5) / 127.5
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format




autoencoder = load_model("conv_autoenoder_1.h5");

encoded_input = Input(shape=(4, 4, 8))
deco = autoencoder.layers[-7](encoded_input)
deco = autoencoder.layers[-6](deco)
deco = autoencoder.layers[-5](deco)
deco = autoencoder.layers[-4](deco)
deco = autoencoder.layers[-3](deco)
deco = autoencoder.layers[-2](deco)
deco = autoencoder.layers[-1](deco)
decoder = Model(encoded_input, deco)


image_shape = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format
enco = autoencoder.layers[0](image_shape)
enco = autoencoder.layers[1](enco)
enco = autoencoder.layers[2](enco)
enco = autoencoder.layers[3](enco)
enco = autoencoder.layers[4](enco)
enco = autoencoder.layers[5](enco)
enco = autoencoder.layers[6](enco)
encoder = Model(image_shape, enco)



# generate only encoded representation of the images - for future use in other algorithms
encoded_imgs = encoder.predict(x_test)

# using full autoencoder
# decoded_imgs = autoencoder.predict(x_test)

# using decoder built from some layers of autoencoder
decoded_imgs = decoder.predict(encoded_imgs)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i+1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

n = 10
plt.figure(figsize=(20, 8))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(encoded_imgs[i].reshape(4, 4 * 8).T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
