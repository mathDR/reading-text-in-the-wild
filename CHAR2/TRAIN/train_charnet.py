'''
Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python main.py
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import model_from_json
from keras.callbacks import EarlyStopping
from keras.layers.core import Dense
from keras.utils import np_utils
from keras.optimizers import SGD
import h5py

from ../build_model import build_model
from data_generator import CharnetSampleGenerator

output_str = '0123456789abcdefghijklmnopqrstuvwxyz '
output = [x for x in output_str]
L = len(output)

if __name__ == '__main__':
    batch_size = 1024
    nb_epoch = 200

    # Build model 
    model = build_model()
    # Add multilabel classifier (max length word is 23.  Each word has label in 
    # 0-9a-z' ', so 10+26+1 = 37)
    model.add(Dense(23*37, activation='sigmoid'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd)

    # Load data

    # Train model
    datagen = CharnetSampleGenerator()

    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    model.fit_generator(datagen.flow(batch_size=batch_size), nb_epoch=nb_epoch, 
                        samples_per_epoch = 10000,verbose=1,callbacks=[early_stopping])
    assert False
    json_string = model.to_json()
    open('charnet_trained_architecture.json', 'w').write(json_string)
    model.save_weights('charnet_trained_weights.h5')
