import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.layers import AveragePooling2D, ReLU, Softmax
from tensorflow.keras.regularizers import l2

from qkeras.autoqkeras import *
from qkeras import *
from qkeras.utils import *

# define model
def resnet_v1_eembc():
    # Resnet parameters
    input_shape=[32,32,3] # default size for cifar10
    num_classes=10 # default class number for cifar10
    num_filters = 16 # this should be 64 for an official resnet model

    # Input layer, change kernel size to 7x7 and strides to 2 for an official resnet
    input = Input(shape=input_shape)
    x = QActivation("quantized_bits_featuremap(bits=16,integer=16,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(input)

    x = QConv2DBatchnorm(num_filters, # conv2d
                kernel_size=3,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4),
                kernel_quantizer="quantized_bits(16,16,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(31,31,1,1,alpha='auto')")(x)
    x = ReLU()(x)
    x = QActivation(f"quantized_bits_featuremap(bits=8,integer=8,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    # First stack
    
    # Weight layers
    y = QConv2DBatchnorm(num_filters, # conv2d_1
                kernel_size=3,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4),
                kernel_quantizer="quantized_bits(8,8,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(31,31,1,1,alpha='auto')")(x)
    y = ReLU()(y)
    y = QActivation(f"quantized_bits_featuremap(bits=8,integer=8,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(y)

    y = QConv2DBatchnorm(num_filters, # conv2d_2
                kernel_size=3,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4),
                kernel_quantizer="quantized_bits(16,16,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(31,31,1,1,alpha='auto')")(y)
    y = QActivation(f"quantized_bits_featuremap(bits=8,integer=8,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(y)

    # Overall residual, connect weight layer and identity paths
    x = tf.keras.layers.add([x, y])
    x = ReLU()(x)
    x = QActivation(f"quantized_bits_featuremap(bits=8,integer=8,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)
    
    # Second stack

    # Weight layers
    num_filters = 32 # Filters need to be double for each stack
    y = QConv2DBatchnorm(num_filters, # conv2d_3
                kernel_size=3,
                strides=2,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4),
                kernel_quantizer="quantized_bits(8,8,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(16,16,1,1,alpha='auto')")(x)
    y = ReLU()(y)
    y = QActivation(f"quantized_bits_featuremap(bits=8,integer=8,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(y)

    y = QConv2DBatchnorm(num_filters, # conv2d_4
                kernel_size=3,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4),
                kernel_quantizer="quantized_bits(8,8,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(16,16,1,1,alpha='auto')")(y)
    y = QActivation(f"quantized_bits_featuremap(bits=8,integer=8,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(y)

    # Adjust for change in dimension due to stride in identity
    x = QConv2D(num_filters, # conv2d_5
                kernel_size=1,
                strides=2,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4),
                kernel_quantizer="quantized_bits(8,8,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(16,16,1,1,alpha='auto')")(x)
    x = QActivation(f"quantized_bits_featuremap(bits=8,integer=8,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    # Overall residual, connect weight layer and identity paths
    x = tf.keras.layers.add([x, y])
    x = ReLU()(x)
    x = QActivation(f"quantized_bits_featuremap(bits=8,integer=8,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    # Third stack

    # Weight layers
    num_filters = 64
    y = QConv2DBatchnorm(num_filters, # conv2d_6
                kernel_size=3,
                strides=2,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4),
                kernel_quantizer="quantized_bits(8,8,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(16,16,1,1,alpha='auto')")(x)
    y = ReLU()(y)
    y = QActivation(f"quantized_bits_featuremap(bits=8,integer=8,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(y)
    
    y = QConv2DBatchnorm(num_filters, # conv2d_7
                kernel_size=3,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4),
                kernel_quantizer="quantized_bits(4,4,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(31,31,1,1,alpha='auto')")(y)
    y = QActivation(f"quantized_bits_featuremap(bits=4,integer=4,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(y)
    
    # Adjust for change in dimension due to stride in identity
    x = QConv2D(num_filters, # conv2d_8
                kernel_size=1,
                strides=2,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4),
                kernel_quantizer="quantized_bits(8,8,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(31,31,1,1,alpha='auto')")(x)
    x = QActivation(f"quantized_bits_featuremap(bits=8,integer=8,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)
    
    # Overall residual, connect weight layer and identity paths
    x = tf.keras.layers.add([x, y])
    x = ReLU()(x)
    x = QActivation(f"quantized_bits_featuremap(bits=16,integer=16,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    # Fourth stack
    #   -> NOT PLACED

    # Final classification layer
    pool_size = int(np.amin(x.shape[1:3]))
    x = AveragePooling2D(pool_size=pool_size)(x)
    x = QActivation(f"quantized_bits_featuremap(bits=16,integer=16,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    x = Flatten()(x)

    x = QDense(num_classes, # dense
                kernel_initializer='he_normal',
                kernel_quantizer="quantized_bits(8,8,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(31,31,1,1,alpha='auto')")(x)
    x = QActivation(f"quantized_bits_featuremap(bits=16,integer=16,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)
    output = Softmax()(x)

    # Instantiate model.
    model = Model(inputs=input, outputs=output)
    return model

