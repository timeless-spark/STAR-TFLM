'''
MobilnetV1 from Silicon Labs github page:
https://github.com/SiliconLabs/platform_ml_models/blob/master/eembc/Person_detection/mobilenet_v1_eembc.py
'''

import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, ReLU, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, AveragePooling2D, MaxPooling2D, Softmax
from tensorflow.keras.regularizers import l2

from qkeras.autoqkeras import *
from qkeras import *
from qkeras.utils import *

# define model
def get_model():
    # Mobilenet parameters
    input_shape = [96,96,3] # resized to 96x96 per EEMBC requirement
    num_classes = 2 # person and non-person
    num_filters = 8 # normally 32, but running with alpha=.25 per EEMBC requirement

    input = Input(shape=input_shape)
    x = QActivation("quantized_bits_featuremap(bits=4,integer=4,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(input)

    # 1st layer, pure conv
    # Keras 2.2 model has padding='valid' and disables bias
    x = QConv2DBatchnorm(num_filters,
                kernel_size=3,
                strides=2,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4),
                kernel_quantizer="quantized_bits(4,4,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(31,31,1,1,alpha='auto')")(x)
    x = ReLU()(x)
    x = QActivation("quantized_bits_featuremap(bits=16,integer=16,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    # 2nd layer, depthwise separable conv
    # Filter size is always doubled before the pointwise conv
    # Keras uses ZeroPadding2D() and padding='valid'
    x = QDepthwiseConv2DBatchnorm(kernel_size=3,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4),
                depthwise_quantizer="quantized_bits(16,16,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(31,31,1,1,alpha='auto')")(x)
    x = ReLU()(x)
    x = QActivation("quantized_bits_featuremap(bits=8,integer=8,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    num_filters = 2*num_filters
    x = QConv2DBatchnorm(num_filters,
                kernel_size=1,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4),
                kernel_quantizer="quantized_bits(16,16,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(31,31,1,1,alpha='auto')")(x)
    x = ReLU()(x)
    x = QActivation("quantized_bits_featuremap(bits=8,integer=8,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    # 3rd layer, depthwise separable conv
    x = QDepthwiseConv2DBatchnorm(kernel_size=3,
                strides=2,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4),
                depthwise_quantizer="quantized_bits(8,8,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(31,31,1,1,alpha='auto')")(x)
    x = ReLU()(x)
    x = QActivation("quantized_bits_featuremap(bits=8,integer=8,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    num_filters = 2*num_filters
    x = QConv2DBatchnorm(num_filters,
                kernel_size=1,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4),
                kernel_quantizer="quantized_bits(16,16,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(16,16,1,1,alpha='auto')")(x)
    x = ReLU()(x)
    x = QActivation("quantized_bits_featuremap(bits=16,integer=16,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    # 4th layer, depthwise separable conv
    x = QDepthwiseConv2DBatchnorm(kernel_size=3,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4),
                depthwise_quantizer="quantized_bits(4,4,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(31,31,1,1,alpha='auto')")(x)
    x = ReLU()(x)
    x = QActivation("quantized_bits_featuremap(bits=4,integer=4,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    x = QConv2DBatchnorm(num_filters,
                kernel_size=1,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4),
                kernel_quantizer="quantized_bits(4,4,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(31,31,1,1,alpha='auto')")(x)
    x = ReLU()(x)
    x = QActivation("quantized_bits_featuremap(bits=8,integer=8,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    # 5th layer, depthwise separable conv
    x = QDepthwiseConv2DBatchnorm(kernel_size=3,
                strides=2,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4),
                depthwise_quantizer="quantized_bits(4,4,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(16,16,1,1,alpha='auto')")(x)
    x = ReLU()(x)
    x = QActivation("quantized_bits_featuremap(bits=4,integer=4,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    num_filters = 2*num_filters
    x = QConv2DBatchnorm(num_filters,
                kernel_size=1,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4),
                kernel_quantizer="quantized_bits(8,8,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(31,31,1,1,alpha='auto')")(x)
    x = ReLU()(x)
    x = QActivation("quantized_bits_featuremap(bits=8,integer=8,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    # 6th layer, depthwise separable conv
    x = QDepthwiseConv2DBatchnorm(kernel_size=3,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4),
                depthwise_quantizer="quantized_bits(8,8,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(16,16,1,1,alpha='auto')")(x)
    x = ReLU()(x)
    x = QActivation("quantized_bits_featuremap(bits=16,integer=16,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    x = QConv2DBatchnorm(num_filters,
                kernel_size=1,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4),
                kernel_quantizer="quantized_bits(8,8,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(16,16,1,1,alpha='auto')")(x)
    x = ReLU()(x)
    x = QActivation("quantized_bits_featuremap(bits=8,integer=8,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    # 7th layer, depthwise separable conv
    x = QDepthwiseConv2DBatchnorm(kernel_size=3,
                strides=2,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4),
                depthwise_quantizer="quantized_bits(8,8,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(31,31,1,1,alpha='auto')")(x)
    x = ReLU()(x)
    x = QActivation("quantized_bits_featuremap(bits=4,integer=4,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    num_filters = 2*num_filters
    x = QConv2DBatchnorm(num_filters,
                kernel_size=1,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4),
                kernel_quantizer="quantized_bits(16,16,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(31,31,1,1,alpha='auto')")(x)
    x = ReLU()(x)
    x = QActivation("quantized_bits_featuremap(bits=4,integer=4,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    # 8th
    x = QDepthwiseConv2DBatchnorm(kernel_size=3,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4),
                depthwise_quantizer="quantized_bits(8,8,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(31,31,1,1,alpha='auto')")(x)
    x = ReLU()(x)
    x = QActivation("quantized_bits_featuremap(bits=4,integer=4,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    x = QConv2DBatchnorm(num_filters,
                kernel_size=1,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4),
                kernel_quantizer="quantized_bits(4,4,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(31,31,1,1,alpha='auto')")(x)
    x = ReLU()(x)
    x = QActivation("quantized_bits_featuremap(bits=16,integer=16,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    # 9th
    x = QDepthwiseConv2DBatchnorm(kernel_size=3,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4),
                depthwise_quantizer="quantized_bits(8,8,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(31,31,1,1,alpha='auto')")(x)
    x = ReLU()(x)
    x = QActivation("quantized_bits_featuremap(bits=8,integer=8,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    x = QConv2DBatchnorm(num_filters,
                kernel_size=1,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4),
                kernel_quantizer="quantized_bits(8,8,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(16,16,1,1,alpha='auto')")(x)
    x = ReLU()(x)
    x = QActivation("quantized_bits_featuremap(bits=16,integer=16,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    # 10th
    x = QDepthwiseConv2DBatchnorm(kernel_size=3,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4),
                depthwise_quantizer="quantized_bits(16,16,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(16,16,1,1,alpha='auto')")(x)
    x = ReLU()(x)
    x = QActivation("quantized_bits_featuremap(bits=8,integer=8,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    x = QConv2DBatchnorm(num_filters,
                kernel_size=1,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4),
                kernel_quantizer="quantized_bits(4,4,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(16,16,1,1,alpha='auto')")(x)
    x = ReLU()(x)
    x = QActivation("quantized_bits_featuremap(bits=4,integer=4,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    # 11th
    x = QDepthwiseConv2DBatchnorm(kernel_size=3,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4),
                depthwise_quantizer="quantized_bits(16,16,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(31,31,1,1,alpha='auto')")(x)
    x = ReLU()(x)
    x = QActivation("quantized_bits_featuremap(bits=16,integer=16,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    x = QConv2DBatchnorm(num_filters,
                kernel_size=1,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4),
                kernel_quantizer="quantized_bits(4,4,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(16,16,1,1,alpha='auto')")(x)
    x = ReLU()(x)
    x = QActivation("quantized_bits_featuremap(bits=4,integer=4,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    # 12th
    x = QDepthwiseConv2DBatchnorm(kernel_size=3,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4),
                depthwise_quantizer="quantized_bits(8,8,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(16,16,1,1,alpha='auto')")(x)
    x = ReLU()(x)
    x = QActivation("quantized_bits_featuremap(bits=8,integer=8,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    x = QConv2DBatchnorm(num_filters,
                kernel_size=1,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4),
                kernel_quantizer="quantized_bits(8,8,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(16,16,1,1,alpha='auto')")(x)
    x = ReLU()(x)
    x = QActivation("quantized_bits_featuremap(bits=8,integer=8,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    # 13th layer, depthwise separable conv
    x = QDepthwiseConv2DBatchnorm(kernel_size=3,
                strides=2,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4),
                depthwise_quantizer="quantized_bits(8,8,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(16,16,1,1,alpha='auto')")(x)
    x = ReLU()(x)
    x = QActivation("quantized_bits_featuremap(bits=16,integer=16,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    num_filters = 2*num_filters
    x = QConv2DBatchnorm(num_filters,
                kernel_size=1,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4),
                kernel_quantizer="quantized_bits(4,4,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(31,31,1,1,alpha='auto')")(x)
    x = ReLU()(x)
    x = QActivation("quantized_bits_featuremap(bits=16,integer=16,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    # 14th layer, depthwise separable conv
    x = QDepthwiseConv2DBatchnorm(kernel_size=3,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4),
                depthwise_quantizer="quantized_bits(4,4,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(16,16,1,1,alpha='auto')")(x)
    x = ReLU()(x)
    x = QActivation("quantized_bits_featuremap(bits=8,integer=8,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    x = QConv2DBatchnorm(num_filters,
                kernel_size=1,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4),
                kernel_quantizer="quantized_bits(4,4,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(31,31,1,1,alpha='auto')")(x)
    x = ReLU()(x)
    x = QActivation("quantized_bits_featuremap(bits=4,integer=4,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    x = AveragePooling2D(pool_size=x.shape[1:3])(x)
    #x = MaxPooling2D(pool_size=x.shape[1:3])(x)
    x = QActivation("quantized_bits_featuremap(bits=4,integer=4,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    # Keras inserts Dropout() and a pointwise Conv2D() here
    # We are staying with the paper base structure

    # Flatten, FC layer and classify
    x = Flatten()(x)

    x = QDense(num_classes,
                kernel_quantizer="quantized_bits(4,4,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(16,16,1,1,alpha='auto')")(x)
    x = QActivation("quantized_bits_featuremap(bits=4,integer=4,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    output = Softmax()(x)
    
    # Instantiate model.
    model = Model(inputs=input, outputs=output)
    return model