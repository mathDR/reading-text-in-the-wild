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
- Theano 0.8.1
    - [See installation instructions](http://deeplearning.net/software/theano/install.html#install).

**Note**: You should use version 0.8.1 of Theano, it is available at [Theano-0.8.1](https://pypi.python.org/pypi/Theano/0.8.1)

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

------------------
## Update Keras

The models from the paper utilizes a deep neural network consisting of eight weight layers:  five convolutioanl layers 
and three fully connected layers.  The convolutional layers have the following {filter_size,number of filters}: {5,64}, 
{5,128}, {3,256}, {3,512}, {3,512}.  The first two fully connected layers each have 4096 units and the final layer has
either 851 units (charnet: max length of word = 23 and each element is of ['0,...,9,a,...,z,' '] so 23*37) or 88172 units 
(dictnet: the number of words in the dictionary).  The final classification layer is followed by a softmax normalization
layer.  Rectified linear non-linarities follow every hidden layer and all but the fourth convolutional layers are followed
by 2x2 max pooling.  The inputs to the convolutional layers are zero padded to preserve dimensionality (border_mode=same 
in Keras).  The fixed size input to the model is a 32x100 greyscale image with is zero-centered by subtracting the mean
and normalized by dividing by the standard deviation.

The original model was written in Caffe and MatConvNet which treat max pooling different than Theano.  The cascading
pooling layers lead to a layer with a shape having size 8x25x512.  Caffe applied max pooling to this layer results in 
size 4x13x512, but Keras/Theano pooling results in a layer of size 4x12x512.  This is due to Theano NOT pooling over the
last column in the filter.

Therefore, a custom zero padding function was written to solve this issue.  The filter of size 4x25x512 is zero padded to 
make it 4x26x512 (with a column of zeros) which then can be max pooled to the desired shape.

The new class is denoted 
```python
CustomZeroPadding2D()
```
and should be added to 
```python
keras/layers/convolutional.py
```
The file is located in the KERAS_TWEAKS directory and should overwrite the respective keras file.

The new function for Theano is denoted
```python
custom_spatial_2d_padding()
```
and should be added to 
```python
keras/backend/theano_backend.py
```
The file is also located in the KERAS_TWEAKS directory and should overwrite the respective keras file.


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
## Usage

# Run the Models
To use the CHAR+2 model, cd to the CHAR2 folder 
and run 
```python
use_charnet.py
```

Similary for the DICT+2 model, go to the DICT2 folder and run 
```python
use_dictnet.py
```

# Train the Models

Training is done differently for the two models.  For the CHAR+2 model, the full model is trained over the 
SYNTH dataset.  The number of samples per epoch (set at 10000) could be increased depending upon compute resources.
The SYNTH dataset has close to nine million images available for training.


The DICT+2 model is trained via *incremental learning* (as seen in T.Xiao, et.al. "Error-Driven Incremental Learning
in Deep Convolutional Neural Network for Large-Scale Image Classification"  In ACM MM, pages
177–186. ACM, 2014).

Both models are trained via back-propagation over either binary crossentropy with dropout on the fully connected layers
(charnet) or categorical crossentropy with dropout on the fully connected layers (dictnet). Optimization uses 
stochastic gradient descent (SGD).

# TO DO
As this model is going to be ran on an NVIDIA Jetson TK1, (currently) the trained CharNet network 
(~450 Million parameters!) is too big to fit on the platform.  Therefore, some type of compression is needed
to scale the network.

Prelimiary work from the paper: Reducing the Model Order of Deep Neural Networks Using Information Theory, 
by M. Tu, V. Berisha, Y. Cao and J-s Seo http://arxiv.org/pdf/1605.04859v1.pdf is being studied.

