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

#define model
def get_model():
    # Mobilenet parameters
    input_shape = [96,96,3] # resized to 96x96 per EEMBC requirement
    num_classes = 2 # person and non-person
    num_filters = 8 # normally 32, but running with alpha=.25 per EEMBC requirement

    input = Input(shape=input_shape)
    x = Activation("sigmoid")(input)

    # 1st layer, pure conv
    # Keras 2.2 model has padding='valid' and disables bias
    x = Conv2D(num_filters,
                kernel_size=3,
                strides=2,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Activation("sigmoid")(x)

    # 2nd layer, depthwise separable conv
    # Filter size is always doubled before the pointwise conv
    # Keras uses ZeroPadding2D() and padding='valid'
    x = DepthwiseConv2D(kernel_size=3,
                        strides=1,
                        padding='same',
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Activation("sigmoid")(x)

    num_filters = 2*num_filters
    x = Conv2D(num_filters,
                kernel_size=1,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Activation("sigmoid")(x)

    # 3rd layer, depthwise separable conv
    x = DepthwiseConv2D(kernel_size=3,
                        strides=2,
                        padding='same',
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Activation("sigmoid")(x)

    num_filters = 2*num_filters
    x = Conv2D(num_filters,
                kernel_size=1,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Activation("sigmoid")(x)

    # 4th layer, depthwise separable conv
    x = DepthwiseConv2D(kernel_size=3,
                        strides=1,
                        padding='same',
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Activation("sigmoid")(x)

    x = Conv2D(num_filters,
                kernel_size=1,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Activation("sigmoid")(x)

    # 5th layer, depthwise separable conv
    x = DepthwiseConv2D(kernel_size=3,
                        strides=2,
                        padding='same',
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Activation("sigmoid")(x)

    num_filters = 2*num_filters
    x = Conv2D(num_filters,
                kernel_size=1,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Activation("sigmoid")(x)

    # 6th layer, depthwise separable conv
    x = DepthwiseConv2D(kernel_size=3,
                        strides=1,
                        padding='same',
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Activation("sigmoid")(x)

    x = Conv2D(num_filters,
                kernel_size=1,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Activation("sigmoid")(x)

    # 7th layer, depthwise separable conv
    x = DepthwiseConv2D(kernel_size=3,
                        strides=2,
                        padding='same',
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Activation("sigmoid")(x)

    num_filters = 2*num_filters
    x = Conv2D(num_filters,
                kernel_size=1,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Activation("sigmoid")(x)

    # 8th
    x = DepthwiseConv2D(kernel_size=3,
                        strides=1,
                        padding='same',
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Activation("sigmoid")(x)

    x = Conv2D(num_filters,
                kernel_size=1,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Activation("sigmoid")(x)

    # 9th
    x = DepthwiseConv2D(kernel_size=3,
                        strides=1,
                        padding='same',
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Activation("sigmoid")(x)

    x = Conv2D(num_filters,
                kernel_size=1,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Activation("sigmoid")(x)

    # 10th
    x = DepthwiseConv2D(kernel_size=3,
                        strides=1,
                        padding='same',
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Activation("sigmoid")(x)

    x = Conv2D(num_filters,
                kernel_size=1,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Activation("sigmoid")(x)

    # 11th
    x = DepthwiseConv2D(kernel_size=3,
                        strides=1,
                        padding='same',
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Activation("sigmoid")(x)

    x = Conv2D(num_filters,
                kernel_size=1,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Activation("sigmoid")(x)

    # 12th
    x = DepthwiseConv2D(kernel_size=3,
                        strides=1,
                        padding='same',
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Activation("sigmoid")(x)

    x = Conv2D(num_filters,
                kernel_size=1,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Activation("sigmoid")(x)

    # 13th layer, depthwise separable conv
    x = DepthwiseConv2D(kernel_size=3,
                        strides=2,
                        padding='same',
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Activation("sigmoid")(x)

    num_filters = 2*num_filters
    x = Conv2D(num_filters,
                kernel_size=1,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Activation("sigmoid")(x)

    # 14th layer, depthwise separable conv
    x = DepthwiseConv2D(kernel_size=3,
                        strides=1,
                        padding='same',
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Activation("sigmoid")(x)

    x = Conv2D(num_filters,
                kernel_size=1,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Activation("sigmoid")(x)

    x = AveragePooling2D(pool_size=x.shape[1:3])(x)
    #x = MaxPooling2D(pool_size=x.shape[1:3])(x)
    x = Activation("sigmoid")(x)

    # Keras inserts Dropout() and a pointwise Conv2D() here
    # We are staying with the paper base structure

    # Flatten, FC layer and classify
    x = Flatten()(x)

    x = Dense(num_classes)(x)
    x = Activation("sigmoid")(x)

    output = Softmax()(x)
    
    # Instantiate model.
    model = Model(inputs=input, outputs=output)
    return model