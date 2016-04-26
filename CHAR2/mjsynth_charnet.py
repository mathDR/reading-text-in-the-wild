# coding=utf-8
## Modified from https://github.com/tommasolevato/CNN-Classification/blob/master/mjsynth.py
from os.path import isfile

import logging

import numpy as np
import os.path

import matplotlib.image as mpimg
from skimage.transform import resize
np.random.seed(1)

class MJSYNTH_CHARNET():
    classes = []
    
    def __init__(self, which_set, numExamples):
        self.output_char = [x for x in '0123456789abcdefghijklmnopqrstuvwxyz ']
        self.space_hot = [0]*37
        self.space_hot[-1] = 1
        self.one_hot = [0]*37

        self.height = 32
        self.width = 100
        self.examples = []
        self.img_shape = (1, self.height, self.width)
        self.numExamples = numExamples
        self.which_set = which_set

        if which_set == "train":
            self.fileToLoadFrom = "annotation_train.txt"
        elif which_set == "test":
            self.fileToLoadFrom = "annotation_test.txt"
        elif which_set == "valid":
            self.fileToLoadFrom = "annotation_val.txt"
        else:
            raise ValueError("Set not recognized")

        self.datapath = 'LOCATION OF SYNTH 90kDICT32px/ FOLDER'
        self.loadData()

    def findExamples(self):
        with open(self.datapath + self.fileToLoadFrom) as f:
            for line in f:
                exampleClass = line.split(" ")[1].rstrip()
                file = line.split(" ")[0].rstrip()
                try:
                    self.examples.append(file[2:len(file)])
                    if len(self.examples) == self.numExamples:
                        break
                except KeyError:
                    pass

    def findOtherExamplesIfNeeded(self):
        if len(self.examples) < self.numExamples:
            with open(self.datapath + self.fileToLoadFrom) as f:
                for line in f:
                    file = line.split(" ")[0].rstrip()
                    if file not in self.examples:
                            self.examples.append(file[2:len(file)])
                    if len(self.examples) == self.numExamples:
                            break
        assert len(self.examples) == self.numExamples

    def loadData(self):
        self.findExamples()
        self.findOtherExamplesIfNeeded()
        self.loadImages()

    def loadImages(self):
        self.x = np.zeros((len(self.examples), 1, self.height, self.width), dtype=np.float32)
        i = 0
        tmp = []
        for example in self.examples:
            filename = self.datapath + example
            self.x[i, :, :, :] = self.loadImage(filename)
            classLabel = self.loadClassLabel(filename)
            tmp.append(classLabel)
            i += 1
        self.labels = np.array(tmp)
        
    def loadImage(self, filename):
        if not isfile(filename):
            print filename + "does not exist"
        else:
            img = mpimg.imread(filename)
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]) # Convert to greyscale
            im = resize(img, (32,100), order=1, preserve_range=True)
            im = np.array(im,dtype=np.float32) # convert to single precision
            img = (im - np.mean(im)) / ( (np.std(im) + 0.0001) )

            return img
        
    def loadClassLabel(self, filename):
        word = (filename.split("_")[1]).lower()
        #convert the word in the filename to a one-hot vector of length 37*23
        classLabel = []
        for i,c in enumerate(word):
            ind = self.output_char.index(c)
            tmp_hot = self.one_hot[:]
            tmp_hot[ind] = 1
            classLabel.extend(tmp_hot)
        classLabel.extend((23-(i+1))*self.space_hot)
        return classLabel

if __name__ == '__main__':
    z = MJSYNTH_CHARNET("train",10)
    output_char = [x for x in '0123456789abcdefghijklmnopqrstuvwxyz ']
    for j in range(len(z.labels)):
        y = z.labels[j]
        for i in range(23):
            c = np.where(y[i*37:(i+1)*37]==1)[0][0]
            print output_char[c],
        print ''
