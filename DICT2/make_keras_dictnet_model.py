'''
Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python main.py
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, CustomZeroPadding2D
from keras.utils import np_utils
from keras.optimizers import SGD
import h5py

if __name__ == '__main__':

    # Load all saved weights
    all_weights = np.load('matlab_dictnet_weights.npz')

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
    weights = [all_weights['conv1W'],all_weights['conv1b']]
    model.add(Convolution2D(nb_filters[layer1], nb_conv[layer1], nb_conv[layer1],weights=weights,
        input_shape=(1, img_rows, img_cols),activation='relu',border_mode='same'))

    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    # Second layer - border_mode = 'same' preserves dimensionality
    weights = [all_weights['conv2W'],all_weights['conv2b']]
    model.add(Convolution2D(nb_filters[layer2], nb_conv[layer2], nb_conv[layer2],weights=weights,
        border_mode='same',activation='relu',init='glorot_uniform'))

    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    # Third layer (no max pooling per model software) - border_mode = 'same' preserves dimensionality
    weights = [all_weights['conv3W'],all_weights['conv3b']]
    model.add(Convolution2D(nb_filters[layer3], nb_conv[layer3], nb_conv[layer3],weights=weights,
        border_mode='same',activation='relu',init='glorot_uniform'))

    # 3.5 layer - border_mode = 'same' preserves dimensionality
    weights = [all_weights['conv35W'],all_weights['conv35b']]
    model.add(Convolution2D(nb_filters[layer35], nb_conv[layer35], nb_conv[layer35],weights=weights,
        border_mode='same',activation='relu',init='glorot_uniform'))

    # Need to zero pad one column on the right hand size of the output so pooling works 
    model.add(CustomZeroPadding2D(padding=(0,0,0,1)))

    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    # Fourth layer - border_mode = 'same' preserves dimensionality
    weights = [all_weights['conv4W'],all_weights['conv4b']]
    model.add(Convolution2D(nb_filters[layer4], nb_conv[layer4], nb_conv[layer4],weights=weights,
        border_mode='same',activation='relu',init='glorot_uniform'))

    # First Dense layer
    model.add(Flatten())
    weights = [all_weights['dense1W'],all_weights['dense1b']]
    model.add(Dense(4096,activation='relu',weights=weights))
    model.add(Dropout(0.5))

    # Second Dense layer
    weights = [all_weights['dense2W'],all_weights['dense2b']]
    model.add(Dense(4096,activation='relu',weights=weights))
    model.add(Dropout(0.5))
    
    # Classification layer
    weights = [all_weights['classW'],all_weights['classb']]
    model.add(Dense(88172,activation='softmax',weights=weights))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    json_string = model.to_json()
    open('dict2_architecture.json', 'w').write(json_string)
    model.save_weights('dict2_weights.h5')

