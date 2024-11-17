#https://github.com/mlcommons/tiny/blob/master/benchmark/training/keyword_spotting/keras_model.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, BatchNormalization, Dropout, Reshape
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, AveragePooling2D, GlobalAveragePooling2D, ReLU, Softmax
from tensorflow.keras.regularizers import l2
from kws_utils import prepare_model_settings

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
    x = Activation('sigmoid')(input)

    x = Conv2D(filters, (10,4), strides=(2,2), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(rate=0.2)(x)
    x = Activation('sigmoid')(x)

    # First layer of separable depthwise conv2d
    # Separable consists of depthwise conv2d followed by conv2d with 1x1 kernels
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Activation('sigmoid')(x)

    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Activation('sigmoid')(x)

    # Second layer of separable depthwise conv2d
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Activation('sigmoid')(x)

    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Activation('sigmoid')(x)

    # Third layer of separable depthwise conv2d
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Activation('sigmoid')(x)

    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Activation('sigmoid')(x)

    # Fourth layer of separable depthwise conv2dmodel_name = args.model_architecture
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Activation('sigmoid')(x)

    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(rate=0.4)(x)
    x = Activation('sigmoid')(x)

    # Reduce size and apply final softmax
    x = AveragePooling2D(pool_size=final_pool_size)(x)
    x = Activation('sigmoid')(x)

    x = Flatten()(x)

    x = Dense(model_settings['label_count'])(x)
    x = Activation('sigmoid')(x)

    output = Softmax()(x)

    # Instantiate model.
    model = Model(inputs=input, outputs=output)

    return model