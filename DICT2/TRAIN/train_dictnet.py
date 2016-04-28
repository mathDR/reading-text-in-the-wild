'''
Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train_dictnet.py
'''

from __future__ import print_function
import numpy as np

from keras.layers.core import Dense
from keras.utils import np_utils
from keras.optimizers import SGD
import h5py

from build_model import build_model_train, build_model
from mjsynth_dictnet import MJSYNTH_DICTNET


if __name__ == '__main__':
    batch_size = 1024
    nb_epoch = 1
    total_classes = 0
    previous_samples = []

    # Get Data and mapping for this round of training
    nb_classes = 5000
    nb_examples_per_class = 10
    train_data = MJSYNTH_DICTNET("train", nb_classes, nb_examples_per_class, previous_samples)
    total_classes += nb_classes
    
    labels = np_utils.to_categorical(train_data.labels, nb_classes)
    previous_samples = train_data.class_mapping
    nb_samples = (nb_classes * nb_examples_per_class)    
    nb_train = np.int( 0.8 * nb_samples )

    xtrain = train_data.x[:nb_train,:,:,:]
    ytrain = labels[:nb_train,:]
    xtest = train_data.x[nb_train:,:,:,:]
    ytest = labels[nb_train:,:]

    # Build model (except for last softmax layer)
    model = build_model()
    # Classification layer
    model.add(Dense(nb_classes,activation='softmax',init='glorot_uniform'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    # Train the model
    model.fit(xtrain, ytrain, batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, verbose=1, validation_data=(xtest, ytest))
    
    # Save the model
    json_string = model.to_json()
    open('initial_charnet_architecture.json', 'w').write(json_string)
    model.save_weights('initial_charnet_weights.h5')
    np.save('initial_classes',previous_samples)

    for i in range(3+6+1): # Run this loop 3 times (to get to 20000 classes) then 6 times to get to 80000
        print ('Iteration: ',i,' Total Classes = ',total_classes)
        # Get data and mapping for this round of training
        if i < 3:
            nb_classes = 5000
        elif i < 9:
            nb_classes = 10000
            batch_size = 2048   # Needs to be ~1/5 of total classes per Jaderberg paper
        else:
            nb_classes = 8172
 
        train_data = MJSYNTH_DICTNET("train", nb_classes, nb_examples_per_class, previous_samples)

        labels = np_utils.to_categorical(train_data.labels, total_classes+nb_classes)
        these_samples = train_data.class_mapping
        nb_samples = (nb_classes * nb_examples_per_class)    
        nb_train = np.int( 0.8 * nb_samples )

        xtrain = train_data.x[:nb_train,:,:,:]
        ytrain = labels[:nb_train,:]
        xtest = train_data.x[nb_train:,:,:,:]
        ytest = labels[nb_train:,:]

        # Save the mapping for this iteration of classes
        previous_samples = these_samples + previous_samples # Need to prepend new samples 
        # Build a new model with nb_classes more classes and initialize it with the previous weights
        model2 = build_model_train(previous_model=model)

        # Classification layer
        model2.add(Dense(total_classes+nb_classes,activation='softmax',init='glorot_uniform'))

        # Overwrite the respective weights for previously trained softmax
        weights = model2.layers[-1].get_weights()    # Has shape (4096 x total_classes+nb_classes)
        old_weights = model.layers[-1].get_weights() # Has shape (4096 x total_classes)

        weights[0][:,-total_classes:] = old_weights[0] # Overwrite such that the first nb_classes cols are random
        weights[1][-total_classes:] = old_weights[1]

        model2.layers[-1].set_weights(weights)
        total_classes += nb_classes

        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model2.compile(loss='categorical_crossentropy', optimizer=sgd)

        model2.fit(xtrain, ytrain, batch_size=batch_size, nb_epoch=nb_epoch,
              show_accuracy=True, verbose=1, validation_data=(xtest, ytest))

        # Save the model and weights
        json_string = model2.to_json()
        open('charnet_architecture_'+str(total_classes)+'.json', 'w').write(json_string)
        model2.save_weights('charnet_weights_'+str(total_classes)+'.h5')
        np.save('classes_'+str(total_classes),these_samples)

        # Iterate models
        model = model2

    # Save the mapping built from this training script to lexicon.txt
    np.save('lexicon_mapping',previous_samples)

