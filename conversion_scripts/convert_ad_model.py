import tensorflow as tf
import sys
import qkeras
from qkeras.autoqkeras import *
from qkeras import *
from qkeras.utils import *
import utils
import models.ad_model
import models.ad_model_MPQ
import ad_utils
import glob
from sklearn import metrics

bits = "int4" # "int8", "int16", "MPQ"
star = True # False

from QKerasToTFLM.QKerasToTFLM import Qkeras_to_TFLM as Q2TFLM_standard 
from QKerasToTFLM.QKerasToTFLM_STAR import Qkeras_to_TFLM as Q2TFLM_star

OUT_DATA_DIR = "../esp/socs/profpga-xc7v2000t/test_data"
OUT_MODEL_DIR = "../esp/socs/profpga-xc7v2000t/model_data"

FLATBUFF_DIR = "../flatbuffers"
SCHEMA_FILE = "../tflite-micro/tensorflow/lite/schema/schema.fbs"

WEIGTHS_DIR = "./models/ckp/ad/"

DATASET_DIR = "/opt/mlperftiny-dataset/ad/dev_data/ToyCar"
TEST_DIR = DATASET_DIR + "/dev_data_perf/ToyCar/test"

model_name = "anomaly"

#Load model and quantized weigths
param = ad_utils.yaml_load("/home/manca/TinyMLPerf-Inference/scripts/baseline.yaml")
umodel = ad_model.get_model(param["feature"]["n_mels"] * param["feature"]["frames"])

if bits == "MPQ":
    qmodel = ad_model_MPQ.get_model(param["feature"]["n_mels"] * param["feature"]["frames"])
else:
    qmodel = utils.flat_quantize(umodel,int(bits))

qmodel.load_weights(WEIGTHS_DIR + bits + "/ckp.h5", by_name=False)

qmodel.compile(**param["fit"]["compile"])

###################################################################################

dirs = sorted(glob.glob(os.path.abspath("{base}/*".format(base=param["dev_perf_directory"]))))
data_and_y_true = []
for _, target_dir in enumerate(dirs):
    machine_id_list = ad_utils.get_machine_id_list_for_test(target_dir)
    for id_str in machine_id_list:
        test_files, y_true = ad_utils.test_file_list_generator(target_dir, id_str)
        for file_idx, file_path in enumerate(test_files):
            data = ad_utils.file_to_vector_array(file_path,
                                            n_mels=param["feature"]["n_mels"],
                                            frames=param["feature"]["frames"],
                                            n_fft=param["feature"]["n_fft"],
                                            hop_length=param["feature"]["hop_length"],
                                            power=param["feature"]["power"])
            data_and_y_true.append((y_true[file_idx], data))

data_and_y_true = data_and_y_true[::8]
n_samples = len(data_and_y_true)*196

test_data = data_and_y_true[0][1]
for (_, data_i) in data_and_y_true[1:]:
    test_data = np.concatenate((test_data, data_i))

in_nb = qmodel.layers[1].quantizer.bits
in_n_parall = 32 // in_nb

in_shape = [dim for dim in test_data[0,:].shape]
if star :
    reminder = in_shape[0] % in_n_parall
    if reminder != 0:
        in_shape[0] += in_n_parall - reminder

max_idx = in_shape[0]

qmodel(np.asarray(test_data[0]).reshape((1,640)))
    
###################################################################################

print("exporting test data: ")

if star :
    if bits == "MPQ":
        f = open(f"{OUT_DATA_DIR}/star/MPQ" + "/" + model_name + "_test_data.h", "w")
    else:
        f = open(f"{OUT_DATA_DIR}/star/int" + bits + "/" + model_name + "_test_data.h", "w")
else :
    f = open(f"{OUT_DATA_DIR}/standard/int" + bits + "/" + model_name + "_test_data.h", "w")

f.write(f"struct data_and_lable {{\n\tconst float data[{max_idx}];\n}};\n\n")
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

    in_buffer = test_data[i,:].flatten().tolist()

    if star :
        if reminder != 0:
            insert_pos = in_shape[0]
            for miss_0 in range(in_n_parall-reminder):
                in_buffer.insert(insert_pos, 0)
    
    f.write(f"\t{{\n\t\t.data = {{{', '.join(map(str,in_buffer))}}}}}{',' if i != (test_data.shape[0]-1) else ''}\n")

f.write("};")
f.close()

print("DONE exporting test data\n")

##################################################################################

print("doing calibration: \n")

file_list_train = sorted(glob.glob(os.path.join(DATASET_DIR,"train/*.wav")))
if (len(file_list_train)==0):
    raise Exception("No .wav data")

train_data = ad_utils.list_to_vector_array(file_list_train,n_mels=128,frames=5,n_fft=1024,hop_length=512,power=2.0)
np.random.shuffle(train_data)

n_calibr_sample = 2000

alpha_of_list = {}
beta_of_list = {}

for iter, x in enumerate(train_data[0:n_calibr_sample]):
    
    data = x.reshape(1, x.shape[0])

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
        
        layer.quantizer.scale1 = tf.convert_to_tensor(scale_of.reshape(layer.quantizer.scale1.numpy().shape), dtype=tf.float32)
        layer.quantizer.zeropoint = tf.convert_to_tensor(z_of.reshape(layer.quantizer.zeropoint.numpy().shape), dtype=tf.float32)

        layer.quantizer.set_training_flag(flag=False)

print("DONE doing calibration\n")

##################################################################################

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

##################################################################################

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

