# coding=utf-8
## Modified from https://github.com/tommasolevato/CNN-Classification/blob/master/mjsynth.py
from os.path import isfile

import logging

import numpy as np
import os.path

import matplotlib.image as mpimg
from skimage.transform import resize

def unique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

class MJSYNTH_DICTNET():

    def __init__(self, which_set, numOfClasses,
                 numOfExamplesPerClass, excluded_examples):
        # excluded_examples denotes the indicies of words in lexicon.txt that have already been used
        self.height = 32
        self.width = 100
        self.examples = []
        self.excluded_examples = excluded_examples
        self.img_shape = (1, self.height, self.width)
        self.numOfClasses = numOfClasses
        self.numOfExamplesPerClass = numOfExamplesPerClass
        self.examplesPerClassCount = {}
        self.which_set = which_set
        
        if which_set == "train":
            self.fileToLoadFrom = "annotation_train.txt"
        elif which_set == "test":
            self.fileToLoadFrom = "annotation_test.txt"
        elif which_set == "valid":
            self.fileToLoadFrom = "annotation_val.txt"
        else:
            raise ValueError("Set not recognized")

        self.classes = []
        self.class_mapping = []

        self.datapath = 'PATH TO SYNTH FOLDER /mnt/ramdisk/max/90kDICT32px/'
        self.loadData()

    def findClasses(self):
        assert self.numOfClasses <= 88172 # Length of lexicon.txt
        choice_set = list( set(range(0,88171)).difference(set(self.excluded_examples)) )
        N = len(choice_set)
        while len(self.classes) < self.numOfClasses:
            randomClass = choice_set[np.random.randint(0, N)]
            if randomClass not in self.classes:
                self.classes.append(randomClass.__str__())
                self.class_mapping.append(randomClass)
                 
        assert len(self.classes) == self.numOfClasses

    def findExamples(self):
        for classToInitialize in self.classes:
            self.examplesPerClassCount[classToInitialize] = 0

        with open(self.datapath + self.fileToLoadFrom) as f:
            for line in f:
                exampleClass = line.split(" ")[1].rstrip()
                file = line.split(" ")[0].rstrip()
                try:
                    if self.examplesPerClassCount[exampleClass] < self.numOfExamplesPerClass:
                        self.examples.append(file[2:len(file)])
                        self.examplesPerClassCount[exampleClass] += 1
                    if len(self.examples) == self.numOfClasses * self.numOfExamplesPerClass:
                        break
                except KeyError:
                    pass

    def findOtherExamplesIfNeeded(self):
        if len(self.examples) < self.numOfClasses * self.numOfExamplesPerClass:
            with open(self.datapath + self.fileToLoadFrom) as f:
                for line in f:
                    exampleClass = line.split(" ")[1].rstrip()
                    file = line.split(" ")[0].rstrip()
                    if exampleClass in self.classes and file not in self.examples:
                            self.examples.append(file[2:len(file)])
                            self.examplesPerClassCount[exampleClass] += 1
                    if len(self.examples) == self.numOfClasses * self.numOfExamplesPerClass:
                            break
        assert len(self.examples) == self.numOfClasses * self.numOfExamplesPerClass

    def loadData(self):
        if self.classes == []:
            self.findClasses()
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
        classLabelTokens = filename.split("_")
        classLabel = classLabelTokens[-1].split(".")[0]
        assert classLabel in self.classes
        # Now map this classlabel to its respective element in range(0,numClasses)
        tmp = []
        tmp.append(self.class_mapping.index(np.int(classLabel)))
        return tmp

if __name__ == '__main__':
    z = MJSYNTH_DICTNET("train",5,2,[])
    for i,y in enumerate(z.labels):
        print i,y,z.classes[y[0]]
    print z.class_mapping
    z = MJSYNTH_DICTNET("train",3,10,z.class_mapping)
    for i,y in enumerate(z.labels):
        print i,y,z.classes[y[0]]
    print z.class_mapping

    
