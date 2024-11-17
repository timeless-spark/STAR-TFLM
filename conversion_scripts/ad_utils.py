import numpy
import librosa
import sys
import tensorflow as tf
import yaml
import os
import glob
import csv
from sklearn import metrics

def yaml_load(file_dir):
    with open(os.path.join(file_dir)) as stream:
        param = yaml.safe_load(stream)
    return param


#file_to_vector function https://github.com/mlcommons/tiny/blob/master/benchmark/training/anomaly_detection/common.py
#@author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
# Copyright (C) 2020 Hitachi, Ltd. All right reserved.
def file_to_vector_array(file_name,
                            n_mels=64,
                            frames=5,
                            n_fft=1024,
                            hop_length=512,
                            power=2.0):
    """
    convert file_name to a vector array.
    file_name : str
        target .wav file
    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 generate melspectrogram
    try:
        y,sr = librosa.load(file_name, sr=None, mono=False)
    except:
        raise Exception("File broken or doesn't exist")

    # 02a generate melspectrogram using librosa
    mel_spectrogram = librosa.feature.melspectrogram(   y=y,
                                                        sr=sr,
                                                        n_fft=n_fft,
                                                        hop_length=hop_length,
                                                        n_mels=n_mels,
                                                        power=power )

    # 03 convert melspectrogram to log mel energy
    log_mel_spectrogram = 20.0 / power * numpy.log10(mel_spectrogram + sys.float_info.epsilon)

    # 3b take central part only
    log_mel_spectrogram = log_mel_spectrogram[:,50:250];

    # 04 calculate total vector size
    vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1

    # 05 skip too short clips
    if vector_array_size < 1:
        return numpy.empty((0, dims))

    # 06 generate feature vectors by concatenating multiframes
    vector_array = numpy.zeros((vector_array_size, dims))
    for t in range(frames):
        vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vector_array_size].T

    return vector_array


#list_to_vector function https://github.com/mlcommons/tiny/blob/master/benchmark/training/anomaly_detection/00_train.py
#@author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
# Copyright (C) 2020 Hitachi, Ltd. All right reserved.
def list_to_vector_array(   file_list,
                            msg="calc...",
                            n_mels=64,
                            frames=5,
                            n_fft=1024,
                            hop_length=512,
                            power=2.0):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.
    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.
    return : numpy.array( numpy.array( float ) )
        vector array for training (this function is not used for test.)
        * dataset.shape = (number of feature vectors, dimensions of feature vectors)
    """
    # calculate the number of dimensions
    dims = n_mels * frames

    # iterate file_to_vector_array()

    for idx in range(len(file_list)):
        vector_array = file_to_vector_array(file_list[idx],
                                            n_mels=n_mels,
                                            frames=frames,
                                            n_fft=n_fft,
                                            hop_length=hop_length,
                                            power=power)
        if idx == 0:
            dataset = numpy.zeros((vector_array.shape[0] * len(file_list), dims), float)
        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array
    return dataset


def test_file_list_generator(   target_dir,
                                id_name,
                                dir_name="test",
                                prefix_normal="normal",
                                prefix_anomaly="anomaly",
                                ext="wav"):
    """
    target_dir : str
        base directory path of the dev_data or eval_data
    id_name : str
        id of wav file in <<test_dir_name>> directory
    dir_name : str (default="test")
        directory containing test data
    prefix_normal : str (default="normal")
        normal directory name
    prefix_anomaly : str (default="anomaly")
        anomaly directory name
    ext : str (default="wav")
        file extension of audio files
    return :
        if the mode is "development":
            test_files : list [ str ]
                file list for test
            test_labels : list [ boolean ]
                label info. list for test
                * normal/anomaly = 0/1
        if the mode is "evaluation":
            test_files : list [ str ]
                file list for test
    """

    normal_files = sorted(glob.glob("{dir}/{dir_name}/{prefix_normal}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                                dir_name=dir_name,
                                                                                                prefix_normal=prefix_normal,
                                                                                                id_name=id_name,
                                                                                                ext=ext)))
    
    normal_labels = numpy.zeros(len(normal_files))
    anomaly_files = sorted(glob.glob("{dir}/{dir_name}/{prefix_anomaly}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                                dir_name=dir_name,
                                                                                                prefix_anomaly=prefix_anomaly,
                                                                                                id_name=id_name,
                                                                                                ext=ext)))
    anomaly_labels = numpy.ones(len(anomaly_files))
    files = numpy.concatenate((normal_files, anomaly_files), axis=0)
    labels = numpy.concatenate((normal_labels, anomaly_labels), axis=0)
    if len(files) == 0:
        raise Exception("No wav file")
    
    return files, labels


def custom_metric(y_true,y_pred):
    """ y_true = X
        y_pred = autoencoder X representation"""
        
    k = tf.reduce_mean(tf.math.square(y_pred - y_true), axis=-1)
    k = k/10  #when loss=9 which is the best result, our metric will be 1
    return 1/(1+k)


def get_machine_id_list_for_test(target_dir,
                                 dir_name="test",
                                 ext="wav"):
    """
    target_dir : str
        base directory path of "dev_data" or "eval_data"
    test_dir_name : str (default="test")
        directory containing test data
    ext : str (default="wav)
        file extension of audio files
    return :
        machine_id_list : list [ str ]
            list of machine IDs extracted from the names of test files
    """
    import re
    import itertools

    # create test files
    dir_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    file_paths = sorted(glob.glob(dir_path))
    # extract id
    machine_id_list = sorted(list(set(itertools.chain.from_iterable(
        [re.findall('id_[0-9][0-9]', ext_id) for ext_id in file_paths]))))
    return machine_id_list


def save_csv(save_file_path,
             save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)


def test(model, weight_dir_base, ckp, load_pretrained, param, mode=1, perf_full=0):

    """
    1) LOOP per target
    2) SPLIT per ID
    3) LOOP per ID 
    """

    if perf_full==0:
        print("Loading dev_data")
        dir_path = os.path.abspath("{base}/*".format(base=param["dev_directory"]))
    else:
        print("Loading dev_data_perf")
        dir_path = os.path.abspath("{base}/*".format(base=param["dev_perf_directory"]))
    dirs = sorted(glob.glob(dir_path))

    # initialize lines in csv for AUC and pAUC
    csv_lines = []

    # loop of the base directory
    for idx, target_dir in enumerate(dirs): #WE ONLY USE TOY CAR
        print("\n===========================")
        print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx+1, total=len(dirs)))
        machine_type = os.path.split(target_dir)[1]

        print("============== MODEL LOAD ==============")

        model.summary()

        if mode:
            # results by type
            csv_lines.append([machine_type])
            csv_lines.append(["id", "AUC", "pAUC"])
            performance = []

        machine_id_list = get_machine_id_list_for_test(target_dir)    

        for id_str in machine_id_list:
            # load test file
            test_files, y_true = test_file_list_generator(target_dir, id_str) 

            # setup anomaly score file path
            anomaly_score_csv = "{result}/anomaly_score_{machine_type}_{id_str}_pretrained{load_pretrained}.csv".format(
                                                                                     result=param["result_directory"],
                                                                                     machine_type=machine_type,
                                                                                     id_str=id_str,
                                                                                     load_pretrained=load_pretrained)
            anomaly_score_list = []

            print("\n============== BEGIN TEST FOR A MACHINE ID ==============")
            y_pred = [0. for k in test_files]
            for file_idx, file_path in enumerate(test_files):          
                try:
                    data = file_to_vector_array(file_path,
                                                    n_mels=param["feature"]["n_mels"],
                                                    frames=param["feature"]["frames"],
                                                    n_fft=param["feature"]["n_fft"],
                                                    hop_length=param["feature"]["hop_length"],
                                                    power=param["feature"]["power"])
                    pred = model.predict(data)
                    errors = numpy.mean(numpy.square(data - pred), axis=1)
                    y_pred[file_idx] = numpy.mean(errors)
                    anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
                except Exception as e:
                    raise Exception("file broken")

            print("anomaly score result ->  {}".format(anomaly_score_csv))

            if mode:
                # append AUC and pAUC to lists
                auc = metrics.roc_auc_score(y_true, y_pred)
                p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=param["max_fpr"])
                csv_lines.append([id_str.split("_", 1)[1], auc, p_auc])
                performance.append([auc, p_auc])
                print("AUC : {}".format(auc))
                print("pAUC : {}".format(p_auc))

            print("\n============ END OF TEST FOR A MACHINE ID ============")

        if mode:
            # calculate averages for AUCs and pAUCs
            averaged_performance = numpy.mean(numpy.array(performance, dtype=float), axis=0)
            csv_lines.append(["Average"] + list(averaged_performance))
            csv_lines.append([])
            print("AUC average : {}".format(averaged_performance[0])) 
            print("pAUC average : {}".format(averaged_performance[1]))

    if mode:
        # output results
        if load_pretrained == 0:
            result_path = "{result}/{file_name}".format(result=param["result_directory"], file_name=param["result_file"])
        else:
            result_path = "{result}/{file_name}".format(result=param["result_directory"], file_name=param["result_file_refmodel"])
        print("AUC and pAUC results -> {}".format(result_path))


def ReturnDatasetAndPrediction(model, param):

    print("Loading dev_data_perf")
    dir_path = "/opt/mlperftiny-dataset/ad/dev_data_perf/ToyCar"
    dirs = sorted(glob.glob(dir_path))

    # loop of the base directory
    for idx, target_dir in enumerate(dirs): #WE ONLY USE TOY CAR

        machine_id_list = get_machine_id_list_for_test(target_dir)

        data_list = []
        pred_list = []

        for id_str in machine_id_list:
            # load test file
            test_files, y_true = test_file_list_generator(target_dir, id_str) 

            for file_idx, file_path in enumerate(test_files):
                try:
                    data = file_to_vector_array(file_path,
                                                    n_mels=param["feature"]["n_mels"],
                                                    frames=param["feature"]["frames"],
                                                    n_fft=param["feature"]["n_fft"],
                                                    hop_length=param["feature"]["hop_length"],
                                                    power=param["feature"]["power"])
                    pred = model.predict(data)
                    for i in range(data.shape[0]):
                        data_list.append(data[i])
                        pred_list.append(pred[i])
                except Exception as e:
                    raise Exception("file broken")
                
    return (data_list, pred_list)
