import tensorflow as tf
import sys
import qkeras
from qkeras.autoqkeras import *
from tensorflow.keras.optimizers import *
from qkeras import *
from qkeras.utils import *
import utils
import models.vww_model
import models.vww_model_MPQ
import numpy as np

bits = "int4" # "int8", "int16", "MPQ"
star = True # False

from QKerasToTFLM.QKerasToTFLM import Qkeras_to_TFLM as Q2TFLM_standard
from QKerasToTFLM.QKerasToTFLM_STAR import Qkeras_to_TFLM as Q2TFLM_star

OUT_DATA_DIR = "../esp/socs/profpga-xc7v2000t/test_data"
OUT_MODEL_DIR = "../esp/socs/profpga-xc7v2000t/model_data"

FLATBUFF_DIR = "../flatbuffers"
SCHEMA_FILE = "../tflite-micro/tensorflow/lite/schema/schema.fbs"

WEIGTHS_DIR = "./models/ckp/vww/"

model_name = "mobilenet"

DATASET_DIR = os.path.join('/opt/mlperftiny-dataset/vw/fdataset')
TEST_DIR = '/opt/mlperftiny-dataset/vw/perf_dataset'

IMAGE_SIZE = 96
BATCH_SIZE = 1
validation_split = 0.1

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=0,
    width_shift_range=0,
    height_shift_range=0,
    zoom_range=0,
    horizontal_flip=False,
    validation_split=validation_split,
    rescale=1. / 255)

train_generator = datagen.flow_from_directory(
    DATASET_DIR,                                
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset='training',
    seed = 5,
    color_mode='rgb')

val_generator = datagen.flow_from_directory(
    DATASET_DIR,                               
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset='validation',
    seed=5,
    color_mode='rgb')

train_data = np.empty([2000,96,96,3])
for i in range(2000):
    train_data[i,:,:,:] = train_generator[i][0]

datagen_test = tf.keras.preprocessing.image.ImageDataGenerator(      
    rotation_range=0,
    width_shift_range=0,
    height_shift_range=0,
    zoom_range=0,
    horizontal_flip=False,
    rescale=1. / 255)

test_generator = datagen_test.flow_from_directory(
    TEST_DIR,                               
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    seed = 5,
    color_mode='rgb')

#Load model and quantized weigths
umodel = vww_model.get_model()
if bits == "MPQ":
    qmodel = vww_model_MPQ.get_model()
else:
    qmodel = utils.flat_quantize(umodel,int(bits))

qmodel.load_weights(WEIGTHS_DIR + bits + "/ckp.h5", by_name=False)

qmodel.compile(optimizer=tf.keras.optimizers.Adam(0.00025),loss='categorical_crossentropy',metrics=['accuracy']) 

n_samples = 100
test_data = np.empty([n_samples,96,96,3])
for i in range(n_samples):
    test_data[i,:,:,:] = test_generator[i*1000//n_samples][0]

lable_person = np.empty([n_samples,]).astype(int)
for i in range(n_samples):
    lable_person[i] = test_generator[i*1000//n_samples][1][0][0].astype(int)

in_nb = qmodel.layers[1].quantizer.bits
in_n_parall = 32 // in_nb

in_shape = [dim for dim in test_data[0,:,:,:].shape]
if star :
    reminder = in_shape[2] % in_n_parall
    original_channels = in_shape[2]
    in_shape[2] += in_n_parall - reminder

max_idx = in_shape[0]*in_shape[1]*in_shape[2]

qmodel(np.asarray(test_data[0]).reshape((1,96,96,3)))

###################################################################################

print("exporting test data: ")

if star :
    if bits == "MPQ":
        f = open(f"{OUT_DATA_DIR}/star/MPQ" + "/" + model_name + "_test_data.h", "w")
    else:
        f = open(f"{OUT_DATA_DIR}/star/int" + bits + "/" + model_name + "_test_data.h", "w")
else :
    f = open(f"{OUT_DATA_DIR}/standard/int" + bits + "/" + model_name + "_test_data.h", "w")

f.write(f"struct data_and_lable {{\n\tconst int person;\n\tconst int non_person;\n\tconst float data[{max_idx}];\n}};\n\n")
f.write(f"extern const struct data_and_lable data_vect[];\n\n")
f.write(f"constexpr int input_size = {max_idx};\n\n")
f.write(f"constexpr int test_samples = {n_samples};")

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
    
    f.write(f"\t{{\n\t\t.person = {lable_person[i]},\n\t\t.data = {{{', '.join(map(str,in_buffer))}}}}}{',' if i != (test_data.shape[0]-1) else ''}\n")

f.write("};")
f.close()

print("DONE exporting test data\n")

###############################################################

print("forcing softmax activation to 16b:")
for layer in qmodel.layers[::-1]:
    if layer.__class__.__name__ in ["QActivation"]:
        layer.quantizer.bits = 16
        layer.quantizer.integer = 16
        break
print("DONE forcing softmax activation to 16b\n")

###############################################################

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

        before_round = ((beta_of_max_abs*alphaq_of - alpha_of_max_abs*betaq_of) / \
                                        (beta_of_max_abs - alpha_of_max_abs))
        z_of = np.asarray([np.around(((beta_of_max_abs*alphaq_of - alpha_of_max_abs*betaq_of) / \
                                        (beta_of_max_abs - alpha_of_max_abs)), 0)], dtype=np.float64)
        z_of[np.isnan(z_of)] = 0
        z_of[np.isinf(z_of)] = 0
        z_of = np.clip(z_of, alphaq_of, betaq_of)
        z_of = z_of[0]
        
        layer.quantizer.scale1 = tf.convert_to_tensor(scale_of.reshape(layer.quantizer.scale1.numpy().shape))
        layer.quantizer.zeropoint = tf.convert_to_tensor(z_of.reshape(layer.quantizer.zeropoint.numpy().shape))

        layer.quantizer.set_training_flag(flag=False)

print("DONE doing calibration\n")

###############################################################

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

###############################################################

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
