'''Fairly basic set of tools for realtime data augmentation on image data.
Can easily be extended to include new transformations,
new preprocessing methods, etc...
'''
from __future__ import absolute_import

import numpy as np
from keras.utils import np_utils
from mjsynth_charnet import MJSYNTH_CHARNET

import threading

class CharnetSampleGenerator(object):
    '''Generate minibatches with
    real-time data augmentation.
    # Arguments
    
    '''
    def __init__(self):
        self.batch_index = 0
        self.total_batches_seen = 0
        self.total_count = 0
        self.lock = threading.Lock()

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, batch_size=32, shuffle=False, seed=None):
        while 1:
            z = MJSYNTH_CHARNET("train",batch_size,self.total_count)
            self.total_count += batch_size        
            self.X = z.x
            self.y = z.labels
            N = self.X.shape[0]

            index_array = np.arange(N)
            if self.batch_index == 0:
                if shuffle:
                    if seed is not None:
                        np.random.seed(seed + self.total_batches_seen)
                    index_array = np.random.permutation(N)

            current_index = (self.batch_index * batch_size) % N
            if N >= current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = N - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def flow(self, batch_size=32, shuffle=False, seed=None,
             save_to_dir=None, save_prefix='', save_format='jpeg'):

        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.reset()

        self.flow_generator = self._flow_index(batch_size, shuffle, seed)
        return self

    def __iter__(self):
        # needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see # http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.flow_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        bX = np.zeros(tuple([current_batch_size] + list(self.X.shape)[1:]))
        for i, j in enumerate(index_array):
            x = self.X[j]
            bX[i] = x
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(bX[i], scale=True)
                img.save(self.save_to_dir + '/' + self.save_prefix + '_' + str(current_index + i) + '.' + self.save_format)
        bY = self.y[index_array]
        return bX, bY

    def __next__(self):
        # for python 3.x.
        return self.next()


if __name__ == '__main__':
    output_char = [x for x in '0123456789abcdefghijklmnopqrstuvwxyz ']
    datagen = CharnetSampleGenerator()
    for e in range(1):
        print 'Epoch', e
        batches = 0
        for X_batch, Y_batch in datagen.flow(batch_size=5):
            print 'Batch: ',batches
            for y in Y_batch:
                for i in range(23):
                    c = np.where(y[i*37:(i+1)*37]==1)[0][0]
                    print output_char[c],
                print ' '            
            
            batches += 1
            if batches >= 3:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break
