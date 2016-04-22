# reading-text-in-the-wild
# A Keras/Theano implementation of "Synthetic Data and Artificial Neural Networks for Natural Scene Text Recognition" 
by M Jaderberg et.al.

------------------

## Installation

This repository uses the following dependencies:

- numpy, scipy
- matplotlib, skimage (for preprocessing images)
- HDF5 and h5py (optional, required if you use model saving/loading functions)
- Optional but recommended if you use CNNs: cuDNN.
- keras 0.3.3
    - [See installation instructions](https://github.com/fchollet/keras/tree/master)
- Theano 0.8.2
    - [See installation instructions](http://deeplearning.net/software/theano/install.html#install).

**Note**: You should use version 0.8.2 of Theano, it is available at [Theano-0.8.2](https://pypi.python.org/pypi/Theano/0.8.2)

**Note**: You should use version 0.3.3 of Keras, it is available at [Keras-0.3.3](https://pypi.python.org/pypi/Keras/0.3.3)


------------------

## Background

This repository implements the models from the following paper:
M. Jaderberg, K. Simpnyan, A. Vedaldi and A. Zisserman. "Synthetic Data and Artificial Neural Networks for 
Natural Scene Text Recognition" Workshop on Deep Learning, NIPS, 2014. 
[paper](http://www.robots.ox.ac.uk/~vgg/publications/2014/Jaderberg14c/jaderberg14c.pdf)

It is made to be ran on an [INVIDIA Jetson TK1](http://www.nvidia.com/object/jetson-tk1-embedded-dev-kit.html) computer 
(hence the restriction to the older version of Theano. The current Theano supports cuDNN v5 which requires 
CUDA 7.0 and the Jetson only supports up to CUDA 6.5).

-----------------
## Datasets and Models
The training data for the networks comes from the [MJSynth dataset](http://www.robots.ox.ac.uk/~vgg/data/text/) and the
models are extracted from the MATLAB models located at [models](http://www.robots.ox.ac.uk/~vgg/research/text/#sec-models)

The weights from the MATLAB models are extracted for conversion to Keras via the files
```python
extract_dictnet_weights.py
```
for the DICT+2 model and
```python
extract_charnet_weights.py
```
for the CHAR+2 model.  These weights are dumped into numpy files
```python
matlab_dictnet_weights.npz
```
and
```python
matlab_charnet_weights.npz
```
respectively.  Then, to build the respective keras models, run
```python
make_keras_dictnet_model.py
```
and
```python
make_keras_charnet_model.py
```
to produce the json architecture file and the hdf5 weights file for use in the respective model.

-----------------
# Usage

Currently the training files are not uploaded (still in progress), but to use the CHAR+2 model, cd to the CHAR2 folder 
and run 
```python
use_charnet.py
```

Similary for the DICT+2 model, go to the DICT2 folder and run 
```python
use_dictnet.py
```
