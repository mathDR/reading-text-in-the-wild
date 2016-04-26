from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, CustomZeroPadding2D
from keras.utils import np_utils

def build_model():
    # input image dimensions
    img_rows, img_cols = 32, 100

    # CNN Architecture from M.Jaderberg et.al. "Synthetic Data and Artificial Neural Networks for Natural Scene 
    # Text Recognition"  Note:  in their paper, they only have four convolutional layers, but their code has five

    # number of convolutional filters to use
    nb_filters = [64,128,256,512,512]
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = [5,5,3,3,3]

    layer1, layer2, layer3, layer35, layer4 = range(5)

    model = Sequential()

    # First layer - border_mode = 'same' preserves dimensionality
    model.add(Convolution2D(nb_filters[layer1], nb_conv[layer1], nb_conv[layer1],
        input_shape=(1, img_rows, img_cols),activation='relu',border_mode='same',init='glorot_uniform'))

    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    # Second layer - border_mode = 'same' preserves dimensionality
    model.add(Convolution2D(nb_filters[layer2], nb_conv[layer2], nb_conv[layer2],
        border_mode='same',activation='relu',init='glorot_uniform'))

    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    # Third layer (no max pooling per model software) - border_mode = 'same' preserves dimensionality
    model.add(Convolution2D(nb_filters[layer3], nb_conv[layer3], nb_conv[layer3],
        border_mode='same',activation='relu',init='glorot_uniform'))

    # 3.5 layer - border_mode = 'same' preserves dimensionality
    model.add(Convolution2D(nb_filters[layer35], nb_conv[layer35], nb_conv[layer35],
        border_mode='same',activation='relu',init='glorot_uniform'))

    # Need to zero pad one column on the right hand size of the output so pooling works 
    model.add(CustomZeroPadding2D(padding=(0,0,0,1)))

    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    # Fourth layer - border_mode = 'same' preserves dimensionality
    model.add(Convolution2D(nb_filters[layer4], nb_conv[layer4], nb_conv[layer4],
        border_mode='same',activation='relu',init='glorot_uniform'))

    # First Dense layer
    model.add(Flatten())
    model.add(Dense(4096,activation='relu',init='glorot_uniform'))
    model.add(Dropout(0.5))

    # Second Dense layer
    model.add(Dense(4096,activation='relu',init='glorot_uniform'))
    model.add(Dropout(0.5))
    
    return model
