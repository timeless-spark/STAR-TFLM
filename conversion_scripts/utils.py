import os 
import json
import tensorflow.compat.v2 as tf
# V2 Behavior is necessary to use TF2 APIs before TF2 is default TF version internally.
tf.enable_v2_behavior()
from tensorflow.keras.optimizers import *
from tensorflow import keras
from contextlib import redirect_stdout
import qkeras
from qkeras.autoqkeras import *
from qkeras import *
from qkeras.utils import *
import glob
import pandas as pd
import subprocess
import numpy as np
from tensorflow.keras.utils import to_categorical
import pickle
from qkeras.utils import model_quantize
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

def flat_quantize(model, bits):
  """
    Flat homogeneous quantization of a floating point model using qkeras.utils.model_quantize
    
      model:        pre-trained floating point model
      bits:         flat quantization bits
      
  """

  kernel_quant = "quantized_bits(%s,%s,1,1,alpha='auto')" % (bits,bits)
  bias_quant = "quantized_bits(31,31,1,1,alpha='auto')"
  act_quant = "quantized_bits_featuremap(%s,%s,1,1,alpha='auto',scale_axis=0)" % (bits,bits)
  
  config = {
    "QActivation": { "sigmoid": act_quant },
    "QConv2D": {
        "kernel_quantizer": kernel_quant,
        "bias_quantizer": bias_quant
    },
    
    "QDepthwiseConv2D": {
          "depthwise_quantizer": kernel_quant,
          "bias_quantizer": bias_quant
    },
    
    "QBatchNormalization": {},
    "QDense": {
      "kernel_quantizer": kernel_quant,
      "bias_quantizer": bias_quant 
    }
  }
  
  qmodel = model_quantize(model,
                    config,
                    bits, #DUMMY
                    transfer_weights=True,
                    enable_bn_folding=True)

  return qmodel

def split_list(list, n):
    splitted_list = []
    for i in range(len(list)//n):
        splitted_list.append(list[n*i:n*i+n])
    return splitted_list

def get_parent_layers(model, layer):

    relevant_nodes = []
    for v in model._nodes_by_depth.values():
        relevant_nodes += v

    connections = []
    for node in layer._inbound_nodes:
        if relevant_nodes and node not in relevant_nodes:
            continue

        for inbound_layer, _, _, _ in node.iterate_inbound():
            connections.append(inbound_layer.name)

    return connections

def net_family_tree(model):
    find_the_parent = {}
    find_the_child = {}
    for layer in model.layers:
        parents = get_parent_layers(model, layer)
        find_the_parent[layer.name] = parents
        for parent in parents:
            if parent in find_the_child:
                find_the_child[parent].append(layer.name)
            else:
                find_the_child[parent] = [layer.name]

    return (find_the_parent, find_the_child)