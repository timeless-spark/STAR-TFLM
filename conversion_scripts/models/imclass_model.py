import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, ReLU,Softmax
from tensorflow.keras.regularizers import l2

#define model
def resnet_v1_eembc():
    # Resnet parameters
    input_shape=[32,32,3] # default size for cifar10
    num_classes=10 # default class number for cifar10
    num_filters = 16 # this should be 64 for an official resnet model

    # Input layer, change kernel size to 7x7 and strides to 2 for an official resnet
    input = Input(shape=input_shape)
    x = Activation('sigmoid')(input)

    x = Conv2D(num_filters,
                kernel_size=3,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Activation('sigmoid')(x)

    # First stack
    
    # Weight layers
    y = Conv2D(num_filters,
                kernel_size=3,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4))(x)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    y = Activation('sigmoid')(y)

    y = Conv2D(num_filters,
                kernel_size=3,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4))(y)
    y = BatchNormalization()(y)
    y = Activation('sigmoid')(y) 

    # Overall residual, connect weight layer and identity paths
    x = tf.keras.layers.add([x, y])
    x = ReLU()(x)
    x = Activation('sigmoid')(x)
    
    # Second stack

    # Weight layers
    num_filters = 32 # Filters need to be double for each stack
    y = Conv2D(num_filters,
                kernel_size=3,
                strides=2,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4))(x)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    y = Activation('sigmoid')(y)

    y = Conv2D(num_filters,
                kernel_size=3,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4))(y)
    y = BatchNormalization()(y)
    y = Activation('sigmoid')(y)

    # Adjust for change in dimension due to stride in identity
    x = Conv2D(num_filters,
                kernel_size=1,
                strides=2,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4))(x)
    x = Activation('sigmoid')(x)

    # Overall residual, connect weight layer and identity paths
    x = tf.keras.layers.add([x, y])
    x = ReLU()(x)
    x = Activation('sigmoid')(x)

    # Third stack

    # Weight layers
    num_filters = 64
    y = Conv2D(num_filters,
                kernel_size=3,
                strides=2,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4))(x)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    y = Activation('sigmoid')(y)
    
    y = Conv2D(num_filters,
                kernel_size=3,
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4))(y)
    y = BatchNormalization()(y)
    y = Activation('sigmoid')(y)
    
    # Adjust for change in dimension due to stride in identity
    x = Conv2D(num_filters,
                kernel_size=1,
                strides=2,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4))(x)
    x = Activation('sigmoid')(x)
    
    # Overall residual, connect weight layer and identity paths
    x = tf.keras.layers.add([x, y])
    x = ReLU()(x)
    x = Activation('sigmoid')(x)

    # Fourth stack.
    # While the paper uses four stacks, for cifar10 that leads to a large increase in complexity for minor benefits
    # Uncomments to use it

#    # Weight layers
#    num_filters = 128
#    y = Conv2D(num_filters,
#                  kernel_size=3,
#                  strides=2,
#                  padding='same',
#                  kernel_initializer='he_normal',
#                  kernel_regularizer=l2(1e-4))(x)
#    y = BatchNormalization()(y)
#    y = ReLU()(y)
#    y = Activation('sigmoid')(y)
#
#    y = Conv2D(num_filters,
#                  kernel_size=3,
#                  strides=1,
#                  padding='same',
#                  kernel_initializer='he_normal',
#                  kernel_regularizer=l2(1e-4))(y)
#    y = BatchNormalization()(y)
#    y = Activation('sigmoid')(y)
#
#    # Adjust for change in dimension due to stride in identity
#    x = Conv2D(num_filters,
#                  kernel_size=1,
#                  strides=2,
#                  padding='same',
#                  kernel_initializer='he_normal',
#                  kernel_regularizer=l2(1e-4))(x)
#    x = Activation('sigmoid')(x)
#
#    # Overall residual, connect weight layer and identity paths
#    x = tf.keras.layers.add([x, y])
#    x = ReLU()(x)
#    x = Activation('sigmoid')(x)

    # Final classification layer.
    pool_size = int(np.amin(x.shape[1:3]))
    x = AveragePooling2D(pool_size=pool_size)(x)
#    x = Activation('sigmoid')(x)

    x = Flatten()(x)

    x = Dense(  num_classes,
                kernel_initializer='he_normal')(x)
    x = Activation('sigmoid')(x)
    output = Softmax()(x)

    # Instantiate model.
    model = Model(inputs=input, outputs=output)
    return model