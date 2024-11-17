
import tensorflow as tf
import sys
import qkeras
from qkeras.autoqkeras import *
from qkeras import *
from qkeras.utils import *
import utils
import models.ic_model
import models.ic_model_MPQ
import ic_utils

bits = "int4" # "int8", "int16", "MPQ"
star = True # False

from QKerasToTFLM.QKerasToTFLM import Qkeras_to_TFLM as Q2TFLM_standard 
from QKerasToTFLM.QKerasToTFLM_STAR import Qkeras_to_TFLM as Q2TFLM_star

OUT_DATA_DIR = "../esp/socs/profpga-xc7v2000t/test_data"
OUT_MODEL_DIR = "../esp/socs/profpga-xc7v2000t/model_data"

FLATBUFF_DIR = "../flatbuffers"
SCHEMA_FILE = "../tflite-micro/tensorflow/lite/schema/schema.fbs"

WEIGTHS_DIR = "./models/ckp/ic/"

model_name = "imclass"

DATASET_DIR = '/opt/mlperftiny-dataset/ic/cifar-10/cifar-10-batches-py'

train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names =  imclass_utils.load_cifar_10_data(DATASET_DIR,n=6) #FULL DATASET
PERF_DIR = "/opt/mlperftiny-dataset/ic/cifar-10" 
_idxs = np.load(PERF_DIR+'/perf_samples_idxs.npy')
test_data = test_data[_idxs]
test_labels = test_labels[_idxs]
test_filenames = test_filenames[_idxs]

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=0,
    width_shift_range=0,
    height_shift_range=0,
    horizontal_flip=False,
    validation_split=0.2
)

#Load model and quantized weigths
umodel = ic_model.resnet_v1_eembc()
if bits == "MPQ":
    qmodel = ic_model_MPQ.resnet_v1_eembc()
else:
    qmodel = utils.flat_quantize(umodel,int(bits))

qmodel.load_weights(WEIGTHS_DIR + bits + "/ckp.h5", by_name=False)

qmodel.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics="accuracy")

label_classes = np.argmax(test_labels,axis=1) 
in_nb = qmodel.layers[1].quantizer.bits
in_n_parall = 32 // in_nb

in_shape = [dim for dim in test_data[0,:,:,:].shape]
if star :
    reminder = in_shape[2] % in_n_parall
    original_channels = in_shape[2]
    in_shape[2] += in_n_parall - reminder

max_idx = in_shape[0]*in_shape[1]*in_shape[2]

qmodel(np.asarray(test_data[0]).reshape((1,32,32,3)))

###################################################################################

print("exporting test data: ")

if star :
    if bits == "MPQ":
        f = open(f"{OUT_DATA_DIR}/star/MPQ" + "/" + model_name + "_test_data.h", "w")
    else:
        f = open(f"{OUT_DATA_DIR}/star/int" + bits + "/" + model_name + "_test_data.h", "w")
else :
    f = open(f"{OUT_DATA_DIR}/standard/int" + bits + "/" + model_name + "_test_data.h", "w")

f.write(f"struct data_and_lable {{\n\tconst int lable;\n\tconst float data[{max_idx}];\n}};\n\n")
f.write(f"extern const struct data_and_lable data_vect[];\n\n")
f.write(f"constexpr int input_size = {max_idx};\n\n")
f.write(f"constexpr int test_samples = {test_data.shape[0]};")

f.close()

if star :
    if bits == "MPQ":
        f = open(f"{OUT_DATA_DIR}/star/MPQ" + "/" + model_name + "_test_data.c", "w")
    else:
        f = open(f"{OUT_DATA_DIR}/star/int" + bits + "/" + model_name + "_test_data.c", "w")
else :
    f = open(f"{OUT_DATA_DIR}/standard/int" + bits + "/" + model_name + "_test_data.c", "w")

f.write(f"#include \"{model_name}_test_data.h\"\n\n")
f.write("const struct data_and_lable data_vect[] = {\n")

for i in range(test_data.shape[0]):
    in_buffer = test_data[i,:,:,:].flatten().tolist()
    if star :
        if reminder != 0:
            for H_i in range(in_shape[0]):
                for W_i in range(in_shape[1]):
                    insert_pos = H_i*(in_shape[1]*in_shape[2])+W_i*(in_shape[2])+original_channels
                    for miss_0 in range(in_n_parall-reminder):
                        in_buffer.insert(insert_pos, 0)
    
    f.write(f"\t{{\n\t\t.lable = {label_classes[i]},\n\t\t.data = {{{', '.join(map(str,in_buffer))}}}}}{',' if i != (test_data.shape[0]-1) else ''}\n")

f.write("};")

f.close()

print("DONE exporting test data\n")

###################################################################################

print("forcing softmax activation to 16b:")
for layer in qmodel.layers[::-1]: # read the layers from last to first
    if layer.__class__.__name__ in ["QActivation"]:
        # set to 16 bit just the activation for the softmax
        layer.quantizer.bits = 16
        layer.quantizer.integer = 16
        break
print("DONE forcing softmax activation to 16b\n")

###################################################################################

print("doing calibration: \n")

n_calibr_sample = 2000

alpha_of_list = {}
beta_of_list = {}

for iter, x in enumerate(train_data[0:n_calibr_sample]):
    
    data = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])

    qmodel(data)

    for layer in qmodel.layers:

        if layer.__class__.__name__ in ["QActivation"]:

            quantizer_of = layer.quantizer # it is quantized_bits_featuremap
            
            alpha_of = quantizer_of.alpha_f.numpy().flatten().item()
            beta_of = quantizer_of.beta_f.numpy().flatten().item()
            if layer.name not in alpha_of_list:
                alpha_of_list[layer.name] = [alpha_of]
                beta_of_list[layer.name] = [beta_of]
            else:
                alpha_of_list[layer.name].append(alpha_of)
                beta_of_list[layer.name].append(beta_of)

for layer in qmodel.layers:

    if layer.name in alpha_of_list:
        alpha_of_max_abs = np.min(alpha_of_list[layer.name])
        beta_of_max_abs = np.max(beta_of_list[layer.name])
        alpha_of_list[layer.name] = alpha_of_max_abs
        beta_of_list[layer.name] = beta_of_max_abs

for layer in qmodel.layers:

    if layer.name in alpha_of_list:

        quantizer_of = layer.quantizer
        alphaq_of = quantizer_of.alphaq
        betaq_of = quantizer_of.betaq

        alpha_of_max_abs = alpha_of_list[layer.name]
        beta_of_max_abs = beta_of_list[layer.name]

        scale_of = np.asarray([(beta_of_max_abs - alpha_of_max_abs) / \
                                (betaq_of - alphaq_of)], dtype=np.float64)
        scale_of[np.isnan(scale_of)] = 0
        scale_of[np.isinf(scale_of)] = 0
        scale_of = scale_of[0]

        z_of = - np.asarray([np.around(((alpha_of_max_abs*betaq_of - beta_of_max_abs*alphaq_of) / \
                                        (beta_of_max_abs - alpha_of_max_abs)), 0)], dtype=np.float64)
        z_of[np.isnan(z_of)] = 0
        z_of[np.isinf(z_of)] = 0
        z_of = z_of[0]
        
        layer.quantizer.scale1 = tf.convert_to_tensor(scale_of.reshape(layer.quantizer.scale1.numpy().shape), dtype=tf.float32)
        layer.quantizer.zeropoint = tf.convert_to_tensor(z_of.reshape(layer.quantizer.zeropoint.numpy().shape), dtype=tf.float32)

        layer.quantizer.set_training_flag(flag=False)

print("DONE doing calibration\n")

###################################################################################

print("forcing bias: ")

for layer in qmodel.layers:
    if layer.__class__.__name__ in ["QConv2D", "QConv2DBatchnorm", "QDepthwiseConv2D", "QDepthwiseConv2DBatchnorm", "QDense", "QDenseBatchnorm"]:
        (find_the_parent, find_the_child) = utils.net_family_tree(qmodel)

        input_layer = qmodel.get_layer(find_the_parent[layer.name][0])
        while input_layer.__class__.__name__ not in ["QActivation"]:
            input_layer = qmodel.get_layer(find_the_parent[input_layer.name][0])

        in_quantizer = input_layer.quantizer
        scale_in = in_quantizer.scale1.numpy().flatten()
        
        weight_quantizer = layer.get_quantizers()[0]
        scale_w = weight_quantizer.scale1.numpy().flatten() * weight_quantizer.m_i.numpy().flatten()

        bias_quantizer = layer.get_quantizers()[1]

        bias_quantizer.bits = 32
        bias_quantizer.integer = 32

        scale_b = (scale_w * scale_in) / K.pow(2.0, K.cast_to_floatx(32))

        bias_quantizer.scale1 = tf.convert_to_tensor(scale_b, dtype=tf.float32)
        bias_quantizer.set_scale_fix_flag(flag=True)

print("DONE forcing bias\n")

###################################################################################

print("converting model: ")
if star:
    if bits == "MPQ":
        conv_dir = OUT_MODEL_DIR + "/star" + "/MPQ"
    else:
        conv_dir = OUT_MODEL_DIR + "/star" + "/int" + bits
    converter = Q2TFLM_star(qkeras_model=qmodel, model_name=model_name, flatbuffer_version=3, 
                                script_out_dir=conv_dir, schema_file=SCHEMA_FILE, flatbuffers_dir=FLATBUFF_DIR)
    converter.ExportFlatbuffer()
else:
    converter = Q2TFLM_standard(qkeras_model=qmodel, model_name=model_name, flatbuffer_version=3, 
                                    script_out_dir= (OUT_MODEL_DIR + "/standard" + "/int" + bits), schema_file=SCHEMA_FILE, flatbuffers_dir=FLATBUFF_DIR)
    converter.ExportFlatbuffer()

print("DONE converting model\n")
