# reading-text-in-the-wild
# A Keras/Theano implementation of "Reading Text in the Wild with Convolutional Neural Networks" by M Jaderberg et.al.


## Guiding principles

- __Modularity.__ A model is understood as a sequence or a graph of standalone, fully-configurable modules that can be plugged together with as little restrictions as possible. In particular, neural layers, cost functions, optimizers, initialization schemes, activation functions, regularization schemes are all standalone modules that you can combine to create new models.

- __Minimalism.__ Each module should be kept short and simple. Every piece of code should be transparent upon first reading. No black magic: it hurts iteration speed and ability to innovate.

- __Easy extensibility.__ New modules are dead simple to add (as new classes and functions), and existing modules provide ample examples. To be able to easily create new modules allows for total expressiveness, making Keras suitable for advanced research.

- __Work with Python__. No separate models configuration files in a declarative format. Models are described in Python code, which is compact, easier to debug, and allows for ease of extensibility.


------------------


## Getting started: 30 seconds to Keras

The core data structure of Keras is a __model__, a way to organize layers. There are two types of models: [`Sequential`](http://keras.io/models/#sequential) and [`Graph`](http://keras.io/models/#graph).

Here's the `Sequential` model (a linear pile of layers):

```python
from keras.models import Sequential

model = Sequential()
```

Stacking layers is as easy as `.add()`:

```python
from keras.layers.core import Dense, Activation

model.add(Dense(output_dim=64, input_dim=100, init="glorot_uniform"))
model.add(Activation("relu"))
model.add(Dense(output_dim=10, init="glorot_uniform"))
model.add(Activation("softmax"))
```

Once your model looks good, configure its learning process with `.compile()`:
```python
model.compile(loss='categorical_crossentropy', optimizer='sgd')
```

If you need to, you can further configure your optimizer. A core principle of Keras is to make things reasonably simple, while allowing the user to be fully in control when they need to (the ultimate control being the easy extensibility of the source code).
```python
from keras.optimizers import SGD
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
```

You can now iterate on your training data in batches:
```python
model.fit(X_train, Y_train, nb_epoch=5, batch_size=32)
```

Alternatively, you can feed batches to your model manually:
```python
model.train_on_batch(X_batch, Y_batch)
```

Evaluate your performance in one line:
```python
objective_score = model.evaluate(X_test, Y_test, batch_size=32)
```

Or generate predictions on new data:
```python
classes = model.predict_classes(X_test, batch_size=32)
proba = model.predict_proba(X_test, batch_size=32)
```

Building a network of LSTMs, a deep CNN, a Neural Turing Machine, a word2vec embedder or any other model is just as fast. The ideas behind deep learning are simple, so why should their implementation be painful?

Have a look at these [starter examples](http://keras.io/examples/).

In the [examples folder](https://github.com/fchollet/keras/tree/master/examples) of the repo, you will find more advanced models: question-answering with memory networks, text generation with stacked LSTMs, neural turing machines, etc.


------------------


## Installation

Keras uses the following dependencies:

- numpy, scipy
- pyyaml
- HDF5 and h5py (optional, required if you use model saving/loading functions)
- Optional but recommended if you use CNNs: cuDNN.

*When using the Theano backend:*

- Theano
    - [See installation instructions](http://deeplearning.net/software/theano/install.html#install).

**Note**: You should use the latest version of Theano, not the PyPI version. Install it with:
```
sudo pip install git+git://github.com/Theano/Theano.git
```

*When using the TensorFlow backend:*

- TensorFlow
    - [See installation instructions](https://github.com/tensorflow/tensorflow#download-and-setup).

To install Keras, `cd` to the Keras folder and run the install command:
```
sudo python setup.py install
```

You can also install Keras from PyPI:
```
sudo pip install keras
```



