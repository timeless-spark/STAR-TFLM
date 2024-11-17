""" https://github.com/mlcommons/tiny/blob/master/benchmark/training/anomaly_detection/keras_model.py """

# from import
import tensorflow.keras as keras
import tensorflow.keras.models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ReLU

from qkeras.autoqkeras import *
from qkeras import *
from qkeras.utils import *

# define model
def get_model(inputDim):
    
    #INPUT
    input = Input(shape=(inputDim,))
    h = QActivation("quantized_bits_featuremap(bits=4,integer=4,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(input)

    #LAYER 1
    h = QDenseBatchnorm(128, # dense
                kernel_quantizer="quantized_bits(4,4,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(16,16,1,1,alpha='auto')")(h)
    h = ReLU()(h)
    h = QActivation(f"quantized_bits_featuremap(bits=16,integer=16,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(h)

    #LAYER 2
    
    h = QDenseBatchnorm(128, # dense_1
                kernel_quantizer="quantized_bits(4,4,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(16,16,1,1,alpha='auto')")(h)
    h = ReLU()(h)
    h = QActivation(f"quantized_bits_featuremap(bits=8,integer=8,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(h)

    #LAYER 3
    
    h = QDenseBatchnorm(128, # dense_2
                kernel_quantizer="quantized_bits(4,4,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(16,16,1,1,alpha='auto')")(h)
    h = ReLU()(h)
    h = QActivation(f"quantized_bits_featuremap(bits=4,integer=4,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(h)

    #LAYER 4
    
    h = QDenseBatchnorm(128, # dense_3
                kernel_quantizer="quantized_bits(4,4,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(31,31,1,1,alpha='auto')")(h)
    h = ReLU()(h)
    h = QActivation(f"quantized_bits_featuremap(bits=4,integer=4,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(h)

    #LAYER 5
    
    h = QDenseBatchnorm(8, # dense_4
                kernel_quantizer="quantized_bits(4,4,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(31,31,1,1,alpha='auto')")(h)
    h = ReLU()(h)
    h = QActivation(f"quantized_bits_featuremap(bits=16,integer=16,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(h)

    #LAYER 6
    
    h = QDenseBatchnorm(128, # dense_5
                kernel_quantizer="quantized_bits(16,16,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(16,16,1,1,alpha='auto')")(h)
    h = ReLU()(h)
    h = QActivation(f"quantized_bits_featuremap(bits=8,integer=8,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(h)

    #LAYER 7
    
    h = QDenseBatchnorm(128, # dense_6
                kernel_quantizer="quantized_bits(4,4,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(31,31,1,1,alpha='auto')")(h)
    h = ReLU()(h)
    h = QActivation(f"quantized_bits_featuremap(bits=4,integer=4,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(h)

    #LAYER 8
    
    h = QDenseBatchnorm(128, # dense_7
                kernel_quantizer="quantized_bits(8,8,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(31,31,1,1,alpha='auto')")(h)
    h = ReLU()(h)
    h = QActivation(f"quantized_bits_featuremap(bits=8,integer=8,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(h)

    #LAYER 9
    
    h = QDenseBatchnorm(128, # dense_8
                kernel_quantizer="quantized_bits(8,8,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(16,16,1,1,alpha='auto')")(h)
    h = ReLU()(h)
    h = QActivation(f"quantized_bits_featuremap(bits=16,integer=16,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(h)

    #OUTPUT
    
    h = QDense(inputDim, # dense_9
                kernel_quantizer="quantized_bits(4,4,1,1,alpha='auto')",
                bias_quantizer="quantized_bits(16,16,1,1,alpha='auto')")(h)
    output = QActivation(f"quantized_bits_featuremap(bits=16,integer=16,symmetric=1,keep_negative=1,alpha='auto',scale_axis=0)")(h)

    return Model(inputs=input, outputs=output)
#########################################################################