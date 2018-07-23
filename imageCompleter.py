
import os

import PIL
import numpy as np


from PIL import Image
import random


import pickle


test_Is = pickle.load(open("test/64.pickle", 'rb'))[:1000] / 256.

import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose, Cropping2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.layers import MaxPooling2D, concatenate, Add
from keras.utils import to_categorical
import keras.backend as K

category = True

import imgaug
import keras.utils

patch_size = 256
from keras.engine.topology import Layer
import network


model = network.nvidia_unet()
loss = []
try:

    model.load_weights("no_scaling_fix_overnight")
except:
    pass
import scipy.misc

from keras.datasets import mnist
digits = mnist.load_data()[0][0][:3000]
import scipy.ndimage

def imageExpander(shape, factor):
    input_ = Input(shape)
    
    vals = np.linspace(0, 1, factor + 1)[1:]
    vals = np.append(vals, [np.flip(vals, 0)[1:]])
    vals = vals * np.transpose([vals])
    
    weights = np.zeros((factor * 2 - 1, factor * 2 - 1, shape[-1], shape[-1]))

    for i in range(shape[-1]):
        weights[:, :, i, i] = vals
    
    output = Conv2DTranspose(shape[-1], (factor * 2 - 1, factor * 2 - 1), strides=factor, weights=[weights, np.zeros(shape[-1])] )(input_)
    return Model(input_, output)
    
    
ie = imageExpander((28, 28, 1), 5)
digidt2 = ie.predict(np.expand_dims(digits, -1))[:,:,:,0]
ie = imageExpander((28, 28, 1), 10)
digit3 = ie.predict(np.expand_dims(digits, -1))[:, 30:-30, 30:-30, 0]
digit3.shape



def gram_matrix(input_):
    print(input_)
    reshape_layer = Reshape((int(input_.shape[1] * input_.shape[2]), int(input_.shape[3])))
    flatten = reshape_layer(input_)
    K_n = 1 / int(input_.shape[1] * input_.shape[2]* input_.shape[3])
    flatten_T = Lambda(lambda a: K_n * K.permute_dimensions(a, (0, 2, 1)))(flatten)
    output = keras.layers.dot([flatten, flatten_T], (1, 2))
    return output

lossModel = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(256, 256, 3))

for layer in lossModel.layers:
    layer.trainable=False

selectedlayers = [3, 6, 10]

layers_loss_model = Model(lossModel.inputs, [lossModel.layers[idx].output for idx in selectedlayers] + 
                                            [gram_matrix(lossModel.layers[idx].output) for idx in selectedlayers])

lossModel.layers[1].output

#validation_perceptual_Is = layers_loss_model.predict(test_Is * 256 - 127) + [test_Is]
validation_perceptual_Is = layers_loss_model.predict(test_Is) + [test_Is]

#loss_model_outputs = Lambda(lambda x: x * 256 - 127)(model.output)
loss_model_outputs = layers_loss_model(model.output)

full_model = Model(model.input, loss_model_outputs + model.outputs)

full_model.compile(keras.optimizers.Adam(), ['mean_absolute_error'] * 7, loss_weights=[.05, .05, .05, 120, 120, 120, 9])

import sys
import gc
def makeMask():
    count = 2000
    patch_size = 256
    out = np.zeros((count, patch_size, patch_size))
    smol = digit3[np.random.randint(0, len(digits), count)]
    out[:, :229, :229] = smol / 2
    out = np.roll(out, np.random.randint(0, patch_size), 1)
    out = np.roll(out, np.random.randint(0, patch_size), 2)
    smol = digit3[np.random.randint(0, len(digits), count)]
    out[:, :229, :229] += smol / 2
    out = np.roll(out, np.random.randint(0, patch_size), 1)
    out = np.roll(out, np.random.randint(0, patch_size), 2)
    medium = digidt2[np.random.randint(0, len(digits), count)]
    out[:, :144, :144] += medium / 2
    out = np.roll(out, np.random.randint(0, patch_size), 1)
    out = np.roll(out, np.random.randint(0, patch_size), 2)
    medium = digidt2[np.random.randint(0, len(digits), count)]
    out[:, :144, :144] += medium / 2
    out = np.roll(out, np.random.randint(0, patch_size), 1)
    out = np.roll(out, np.random.randint(0, patch_size), 2)
    mask = np.ones([count, patch_size, patch_size, 3])
    mask -= (np.expand_dims(out > 45, -1))
    return mask

def train():
    count = 2000
    patch_size = 256
    mask = makeMask()

    gc.collect()
    bundle = "bundles/" + random.choice(os.listdir("bundles"))
    Is = pickle.load(open(bundle, 'rb'))[:2000] / 256.
    sample = Is
    print("init")
    try:
        perceptual_Is = layers_loss_model.predict(sample) + [sample]
        #perceptual_Is = layers_loss_model.predict(sample * 256 - 127) + [sample]

        loss.append(full_model.fit([sample, mask], perceptual_Is, epochs=1, 
                                   batch_size=8, 
                                   )
                   )
        
    finally:
        print("del_start")
        print(sys.getrefcount(perceptual_Is[0]))
        del perceptual_Is
        gc.collect()
        print("del end")
    Is = pickle.load(open(bundle, 'rb'))[2000:] / 256.
    sample = Is
    print("init")
    try:
        
        perceptual_Is = layers_loss_model.predict(sample) + [sample]
        #perceptual_Is = layers_loss_model.predict(sample * 256 - 127) + [sample]


        loss.append(full_model.fit([sample, mask], perceptual_Is, epochs=1, 
                                   batch_size=8, 
                                   validation_data=([test_Is, mask[:1000]],
                                                     validation_perceptual_Is))
                   )
        
    finally:
        print("del_start")
        print(sys.getrefcount(perceptual_Is[0]))
        del perceptual_Is
        gc.collect()
        print("del end")
        model.save("no_scaling_fix_overnight")
import gc
perceptual_Is = []
for _ in range(30):
    train()

model.save("no_scaling_fix_overnight")

