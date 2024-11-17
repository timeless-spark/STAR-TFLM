""" https://github.com/mlcommons/tiny/blob/master/benchmark/training/anomaly_detection/keras_model.py """

# from import
import tensorflow.keras as keras
import tensorflow.keras.models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, ReLU, Softmax

# define model
def get_model(inputDim):
    
    #INPUT
    input = Input(shape=(inputDim,))
    h = Activation('sigmoid')(input)

    #LAYER 1
    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = ReLU()(h)
    h = Activation('sigmoid')(h)

    #LAYER 2
    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = ReLU()(h)
    h = Activation('sigmoid')(h)

    #LAYER 3
    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = ReLU()(h)
    h = Activation('sigmoid')(h)

    #LAYER 4
    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = ReLU()(h)
    h = Activation('sigmoid')(h)

    #LAYER 5
    h = Dense(8)(h)
    h = BatchNormalization()(h)
    h = ReLU()(h)
    h = Activation('sigmoid')(h)

    #LAYER 6
    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = ReLU()(h)
    h = Activation('sigmoid')(h)

    #LAYER 7
    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = ReLU()(h)
    h = Activation('sigmoid')(h)

    #LAYER 8
    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = ReLU()(h)
    h = Activation('sigmoid')(h)

    #LAYER 9
    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = ReLU()(h)
    h = Activation('sigmoid')(h)

    #OUTPUT
    h = Dense(inputDim)(h)
    output = Activation('sigmoid')(h)

    return Model(inputs=input, outputs=output)
#########################################################################