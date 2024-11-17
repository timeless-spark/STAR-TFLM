#https://github.com/mlcommons/tiny/blob/master/benchmark/training/keyword_spotting/keras_model.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, BatchNormalization, Dropout, Reshape
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, AveragePooling2D, GlobalAveragePooling2D, ReLU, Softmax
from tensorflow.keras.regularizers import l2
from kws_utils import prepare_model_settings

from qkeras.autoqkeras import *
from qkeras import *
from qkeras.utils import *

# define model
def get_model(args):
    label_count=12
    model_settings = prepare_model_settings(label_count, args)
    
    input_shape = [model_settings['spectrogram_length'], model_settings['dct_coefficient_count'],1]
    filters = 64
    weight_decay = 1e-4
    regularizer = l2(weight_decay)
    final_pool_size = (int(input_shape[0]/2), int(input_shape[1]/2))
    
    # Model layers
    # Input pure conv2d
    input = Input(shape=input_shape)
    x = QActivation("quantized_bits_featuremap(bits=16,integer=16,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(input)

    x = QConv2DBatchnorm(filters, # conv2d
                (10,4), strides=(2,2), padding='same', kernel_regularizer=regularizer,
                kernel_quantizer="quantized_bits(16,16,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(31,31,1,1,alpha='auto')")(x)
    x = ReLU()(x)
    x = QActivation("quantized_bits_featuremap(bits=8,integer=8,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    # First layer of separable depthwise conv2d
    # Separable consists of depthwise conv2d followed by conv2d with 1x1 kernels
    x = QDepthwiseConv2DBatchnorm(depth_multiplier=1, # depthwise_conv2d
                kernel_size=(3,3), padding='same', kernel_regularizer=regularizer,
                depthwise_quantizer="quantized_bits(8,8,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(16,16,1,1,alpha='auto')")(x)
    x = ReLU()(x)
    x = QActivation("quantized_bits_featuremap(bits=8,integer=8,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    x = QConv2DBatchnorm(filters, # conv2d_1
                (1,1), padding='same', kernel_regularizer=regularizer,
                kernel_quantizer="quantized_bits(4,4,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(31,31,1,1,alpha='auto')")(x)
    x = ReLU()(x)
    x = QActivation("quantized_bits_featuremap(bits=4,integer=4,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    # Second layer of separable depthwise conv2d
    x = QDepthwiseConv2DBatchnorm(depth_multiplier=1, # depthwise_conv2d_1
                kernel_size=(3,3), padding='same', kernel_regularizer=regularizer,
                depthwise_quantizer="quantized_bits(8,8,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(16,16,1,1,alpha='auto')")(x)
    x = ReLU()(x)
    x = QActivation("quantized_bits_featuremap(bits=8,integer=8,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    x = QConv2DBatchnorm(filters, # conv2d_2
                (1,1), padding='same', kernel_regularizer=regularizer,
                kernel_quantizer="quantized_bits(4,4,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(16,16,1,1,alpha='auto')")(x)
    x = ReLU()(x)
    x = QActivation("quantized_bits_featuremap(bits=8,integer=8,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    # Third layer of separable depthwise conv2d
    x = QDepthwiseConv2DBatchnorm(depth_multiplier=1, # depthwise_conv2d_2
                kernel_size=(3,3), padding='same', kernel_regularizer=regularizer,
                depthwise_quantizer="quantized_bits(4,4,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(16,16,1,1,alpha='auto')")(x)
    x = ReLU()(x)
    x = QActivation("quantized_bits_featuremap(bits=4,integer=4,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    x = QConv2DBatchnorm(filters, # conv2d_3
                (1,1), padding='same', kernel_regularizer=regularizer,
                kernel_quantizer="quantized_bits(4,4,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(16,16,1,1,alpha='auto')")(x)
    x = ReLU()(x)
    x = QActivation("quantized_bits_featuremap(bits=4,integer=4,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    # Fourth layer of separable depthwise conv2dmodel_name = args.model_architecture
    x = QDepthwiseConv2DBatchnorm(depth_multiplier=1, # depthwise_conv2d_3
                kernel_size=(3,3), padding='same', kernel_regularizer=regularizer,
                depthwise_quantizer="quantized_bits(16,16,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(31,31,1,1,alpha='auto')")(x)
    x = ReLU()(x)
    x = QActivation("quantized_bits_featuremap(bits=4,integer=4,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    x = QConv2DBatchnorm(filters, # conv2d_4
                (1,1), padding='same', kernel_regularizer=regularizer,
                kernel_quantizer="quantized_bits(4,4,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(16,16,1,1,alpha='auto')")(x)
    x = ReLU()(x)
    x = Dropout(rate=0.4)(x)
    x = QActivation("quantized_bits_featuremap(bits=8,integer=8,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    # Reduce size and apply final softmax
    x = AveragePooling2D(pool_size=final_pool_size)(x)
    x = QActivation("quantized_bits_featuremap(bits=16,integer=16,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    x = Flatten()(x)

    x = QDense(model_settings['label_count'], # dense
                kernel_quantizer="quantized_bits(8,8,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(31,31,1,1,alpha='auto')")(x)
    x = QActivation("quantized_bits_featuremap(bits=16,integer=16,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(x)

    output = Softmax()(x)

    # Instantiate model.
    model = Model(inputs=input, outputs=output)

    return model