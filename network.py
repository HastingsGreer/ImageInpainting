import numpy as np

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


import keras.utils
#patch_size=128
patch_size = 256
from keras.engine.topology import Layer
class Bias(Layer):
    def __init__(self, name=""):
        super().__init__()
        
    def build(self, input_shape):
        channel_axis = -1
        self.bias = self.add_weight(shape=(input_shape[-1],),
                                        initializer="zeros",
                                        name='bias',
                                        regularizer=None,
                                        constraint=None)
        super(Bias, self).build(input_shape)
        
    def call(self, inputs):
        outputs = K.bias_add(
                inputs,
                self.bias)
        return outputs
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
keras.layers.Bias = Bias

def partial_convolution(input_, mask, filters, shape, stride, activation):
    convolution_layer = Conv2D(filters, shape, strides=stride, use_bias=False,
                               padding="same")
    
    mask_sum_layer = Conv2D(filters, shape, strides=stride, 
                                  padding="same", 
                                  weights=[np.ones((shape[0], shape[1], input_.shape[-1], filters)),
                                           np.zeros((filters,))])
    
    mask_sum_layer.trainable = False
    
    mask_sum = mask_sum_layer(mask)
    
    new_mask = Lambda(lambda x: K.clip(x, 0, 1))(mask_sum)
    
    output = convolution_layer(keras.layers.multiply([mask , input_]))
    
    inv_sum = Lambda(lambda x: filters * shape[0] * shape[1] / (.0001 + x))(mask_sum) 
    
    output = keras.layers.multiply([output, inv_sum])
    
    output = Bias()(output)
    
    output = activation(output)
    
    return output, new_mask

def full_convolution(input_, mask, filters, shape, stride, activation):
    output = Conv2D(filters, shape, strides=stride, padding="same")(input_)
    output = activation(output)
    return output, "string"

def nvidia_unet():
    input_ = Input((patch_size, patch_size, 3))
    input_mask = Input((patch_size, patch_size, 3))
    skips = []
    output = input_
    mask = input_mask
    for shape, filters in zip([7, 5, 5, 3, 3, 3, 3, 3], [64, 128, 256, 512, 512, 512, 512, 512]):
        skips.append((output, mask))
        print(output.shape)
        output, mask = partial_convolution(output, mask, filters, (shape, shape), 2, 
                                          Activation("relu"))
        if shape != 7:
            output = BatchNormalization()(output)
    for shape, filters in zip([4, 4, 4, 4, 4, 4, 4, 4], [ 512, 512, 512, 512, 256, 128, 64, 3]):
        output = keras.layers.UpSampling2D()(output)
        mask = keras.layers.UpSampling2D()(mask)
        skip_output, skip_mask = skips.pop()
        output = concatenate([output, skip_output], axis=3)
        mask = concatenate([mask, skip_mask], axis=3)
        
        if filters!= 3:
            activation = keras.layers.LeakyReLU(.2)
        else:
            activation = Activation("linear")
        output, mask = partial_convolution(output, mask, filters, (shape, shape), 1, 
                                          activation)
        if filters!= 3:
             output = BatchNormalization()(output)
    assert len(skips) == 0
    return Model([input_, input_mask], [output])


def pad_to_patch_size(image, mask):
    #Network only accepts square images with size as a multiple of 256.
    #Use the mask to make sure that the output only depends on the valid rectangle.
    network_input = np.zeros((1, patch_size, patch_size, 3))
    network_mask = np.zeros((1, patch_size, patch_size, 3))

    network_input[:, :image.shape[1], :image.shape[2]] = image
    network_mask[:, :mask.shape[1], :mask.shape[2]] = mask

    return network_input, network_mask

def set_patch_size_to_fit(z):
    num_patches = (max(z.shape[1], z.shape[2]) - 1) // 256 + 1

    #Sorry that this is global. I promise to fix soon (maybe)
    global patch_size
    patch_size = 256 * num_patches

    

#model = nvidia_unet()
