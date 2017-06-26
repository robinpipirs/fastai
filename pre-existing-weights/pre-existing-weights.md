
# Using existing weights

```If you're after the h5 file for the vgg16 model please browse the link bellow```
```https://drive.google.com/open?id=0B7r0szEM79wwMnJkVDA3Y2RFOVE```

Set up paths.

```python
path = "data/dogscats/"
FILES_PATH = "data/lesson1data/results/"
```

import some stuff


```python
from __future__ import division,print_function

import os, json
from glob import glob
import numpy as np
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt
from numpy.random import random, permutation
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom
```

This shows plots in the web page itself - we always wants to use this when using jupyter notebook:


```python
%matplotlib inline
```

'utils.py' is used to store any little convenience functions we'll want to use.


```python
import utils; reload(utils)
from utils import plots
```

    Using gpu device 0: Tesla K80 (CNMeM is disabled, cuDNN 5103)
    /home/ubuntu/anaconda2/lib/python2.7/site-packages/theano/sandbox/cuda/__init__.py:600: UserWarning: Your cuDNN version is more recent than the one Theano officially supports. If you see any problems, try updating Theano or downgrading cuDNN to version 5.
      warnings.warn(warn)
    Using Theano backend.


Keras wrapper for neural network


```python
import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers import Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop
from keras.preprocessing import image
```

## Model creation

Creating the model involves creating the model architecture, and then loading the model weights into that architecture. We will start by defining the basic pieces of the VGG architecture.

VGG has just one type of convolutional block, and one type of fully connected ('dense') block. Here's the convolutional block definition:


```python
# Import our class, and instantiate
import vgg16; reload(vgg16)
from vgg16 import Vgg16
```


```python
def ConvBlock(layers, model, filters):
    for i in range(layers): 
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(filters, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
```

here's the fully-connected definition.


```python
def FCBlock(model):
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
```

When the VGG model was trained in 2014, the creators subtracted the average of each of the three (R,G,B) channels first, so that the data for each channel had a mean of zero. Furthermore, their software that expected the channels to be in B,G,R order, whereas Python by default uses R,G,B. We need to preprocess our data to make these two changes, so that it is compatible with the VGG model:


```python
# Mean of each channel as provided by VGG researchers
vgg_mean = np.array([123.68, 116.779, 103.939]).reshape((3,1,1))

def vgg_preprocess(x):
    x = x - vgg_mean     # subtract mean
    return x[:, ::-1]    # reverse axis bgr->rgb
```

Now we're ready to define the VGG model architecture - look at how simple it is, now that we have the basic blocks defined!


```python
def VGG_16():
    model = Sequential()
    model.add(Lambda(vgg_preprocess, input_shape=(3,224,224)))

    ConvBlock(2, model, 64)
    ConvBlock(2, model, 128)
    ConvBlock(3, model, 256)
    ConvBlock(3, model, 512)
    ConvBlock(3, model, 512)

    model.add(Flatten())
    FCBlock(model)
    FCBlock(model)
    model.add(Dense(1000, activation='softmax'))
    return model
```

- Convolution layers are for finding patterns in images
- Dense (fully connected) layers are for combining patterns across an image

Now that we've defined the architecture, we can create the model like any python object:


```python
#Instansiate vgg16 model
model = VGG_16()
```

## Loading existing weights

As well as the architecture, we need the weights that the VGG creators trained. The weights are the part of the model that is learnt from the data, whereas the architecture is pre-defined based on the nature of the problem. 
We have fitted the network to cats and dogs therefore we will load that model.  
  
Downloading pre-trained weights is much preferred to training the model ourselves, since otherwise we would have to download the entire Imagenet archive, and train the model for many days! It's very helpful when researchers release their weights, as they did here.


```python
fpath = FILES_PATH+'ft2.h5'
model.load_weights(fpath)
```

##  Getting imagenet predictions

The setup of the imagenet model is now complete, so all we have to do is grab a batch of images and call *predict()* on them.


```python
batch_size = 10
```

Keras provides functionality to create batches of data from directories containing images; all we have to do is to define the size to resize the images to, what type of labels to create, whether to randomly shuffle the images, and how many images to include in each batch. We use this little wrapper to define some helpful defaults appropriate for imagenet data:


```python
def get_batches(dirname, gen=image.ImageDataGenerator(), shuffle=True, 
                batch_size=batch_size, class_mode='categorical'):
    return gen.flow_from_directory(path+dirname, target_size=(224,224), 
                class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)
```

From here we can use exactly the same steps as before to look at predictions from the model.


```python
batches = get_batches('train', batch_size=batch_size)
val_batches = get_batches('valid', batch_size=batch_size)
imgs,labels = next(batches)

# This shows the 'ground truth'
plots(imgs, titles=labels)
```

    Found 23000 images belonging to 2 classes.
    Found 2000 images belonging to 2 classes.



![png](output_30_1.png)



```python
model.predict(imgs, True)
```




    array([[  1.0000e+00,   0.0000e+00],
           [  1.0000e+00,   0.0000e+00],
           [  1.0000e+00,   0.0000e+00],
           [  0.0000e+00,   1.0000e+00],
           [  1.0000e+00,   0.0000e+00],
           [  1.0000e+00,   4.8469e-32],
           [  1.0000e+00,   0.0000e+00],
           [  1.0000e+00,   0.0000e+00],
           [  0.0000e+00,   1.0000e+00],
           [  0.0000e+00,   1.0000e+00]], dtype=float32)




```python

```
