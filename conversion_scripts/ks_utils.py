#functions from: 
#https://github.com/mlcommons/tiny/blob/master/benchmark/training/keyword_spotting/get_dataset.py
#https://github.com/mlcommons/tiny/blob/master/benchmark/training/keyword_spotting/keras_model.py

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.platform import gfile
import numpy as np
import os

#https://github.com/mlcommons/tiny/blob/master/benchmark/training/keyword_spotting/kws_util.py L#8 hardcoded as a python dict
Flags = { 
            "background_volume":0.1,                            #How loud the background noise should be, between 0 and 1
            "background_frequency":0.8,                         #How many of the training samples have background noise mixed in
            "silence_percentage":10.0,                          #How much of the training data should be silence
            "unknown_percentage":10.0,                          #How much of the training data should be unknown words
            "time_shift_ms": 100.0,                             #Range to randomly shift the training audio by in time
            "sample_rate": 16000,                               #Expected sample rate of the wavs
            "clip_duration_ms": 1000,                           #Expected duration in milliseconds of the wavs
            "window_size_ms": 30.0,                             #How long each spectrogram timeslice is
            "window_stride_ms": 20.0,                           #How long each spectrogram timeslice is
            "feature_type": "mfcc",                             #Type of input features. Valid values: "mfcc" (default)
            "dct_coefficient_count":10,                         #How many MFCC or log filterbank energy features
            "epochs": 10,                                       #How many epochs to train was 36
            "num_train_samples": -1,                            #How many samples from the training set to use
            "num_val_samples": -1,                              #How many samples from the validation set to use
            "num_test_samples": -1,                             #How many samples from the test set to use
            "batch_size" :100,                                  #How many items to train with at once
            "num_bin_files": 1000,                              #How many binary test files for benchmark runner to create
            "model_architecture" :"ds_cnn",                     #What model architecture to use                     ????
            "run_test_set": True,                               #In train.py, run model.eval() on test set if True  ????
            "learning_rate": 0.00001,                           #Initial LR
            "lr_sched_name": "step_function"                    #lr schedule scheme name to be picked from lr
}


def prepare_model_settings(label_count, args):
    """Calculates common settings needed for all models.
    Args:
    label_count: How many classes are to be recognized.
    sample_rate: Number of audio samples per second.
    clip_duration_ms: Length of each audio clip to be analyzed.
    window_size_ms: Duration of frequency analysis window.
    window_stride_ms: How far to move in time between frequency windows.
    dct_coefficient_count: Number of frequency bins to use for analysis.
    Returns:
        Dictionary containing common settings.
    """
    
    #( "if feature type==td_samples" was removed because we only enter the "else" since feature type is mfcc)
    
    desired_samples = int(args['sample_rate'] * args['clip_duration_ms'] / 1000)
    dct_coefficient_count = args['dct_coefficient_count']
    window_size_samples = int(args['sample_rate'] * args['window_size_ms'] / 1000)
    window_stride_samples = int(args['sample_rate'] * args['window_stride_ms'] / 1000)
    length_minus_window = (desired_samples - window_size_samples)
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
        fingerprint_size = args['dct_coefficient_count'] * spectrogram_length
    return {
        'desired_samples': desired_samples,
        'window_size_samples': window_size_samples,
        'window_stride_samples': window_stride_samples,
        'feature_type': args['feature_type'], 
        'spectrogram_length': spectrogram_length,
        'dct_coefficient_count': dct_coefficient_count,
        'fingerprint_size': fingerprint_size,
        'label_count': label_count,
        'sample_rate': args['sample_rate'],
        'background_frequency': 0.8,
        'background_volume_range_': 0.1
    }

def convert_dataset(item):
    """Puts the mnist dataset in the format Keras expects, (features, labels)."""
    audio = item['audio']
    label = item['label']
    return audio, label
    
def get_preprocess_audio_func(model_settings,is_training=False,background_data = []):
    def prepare_processing_graph(next_element):
        """Builds a TensorFlow graph to apply the input distortions.
        Creates a graph that loads a WAVE file, decodes it, scales the volume,
        shifts it in time, adds in background noise, calculates a spectrogram, and
        then builds an MFCC fingerprint from that.
        This must be called with an active TensorFlow session running, and it
        creates multiple placeholder inputs, and one output:
            - wav_filename_placeholder_: Filename of the WAV to load.
            - foreground_volume_placeholder_: How loud the main clip should be.
            - time_shift_padding_placeholder_: Where to pad the clip.
            - time_shift_offset_placeholder_: How much to move the clip in time.
            - background_data_placeholder_: PCM sample data for background noise.
            - background_volume_placeholder_: Loudness of mixed-in background.
            - mfcc_: Output 2D fingerprint of processed audio.
        Args:
            model_settings: Information about the current model being trained.
        """
        desired_samples = model_settings['desired_samples']
        background_frequency = model_settings['background_frequency']
        background_volume_range_= model_settings['background_volume_range_']    
        wav_decoder = tf.cast(next_element['audio'], tf.float32)
        wav_decoder = wav_decoder/tf.reduce_max(wav_decoder)

        #Previously, decode_wav was used with desired_samples as the length of array. The
        # default option of this function was to pad zeros if the desired samples are not found
        wav_decoder = tf.pad(wav_decoder,[[0,desired_samples-tf.shape(wav_decoder)[-1]]]) 
        # Allow the audio sample's volume to be adjusted.
        foreground_volume_placeholder_ = tf.constant(1,dtype=tf.float32)

        scaled_foreground = tf.multiply(wav_decoder,
                                        foreground_volume_placeholder_)
        # Shift the sample's start position, and pad any gaps with zeros.
        time_shift_padding_placeholder_ = tf.constant([[2,2]], tf.int32)
        time_shift_offset_placeholder_ = tf.constant([2],tf.int32)
        scaled_foreground.shape
        padded_foreground = tf.pad(scaled_foreground, time_shift_padding_placeholder_, mode='CONSTANT')
        sliced_foreground = tf.slice(padded_foreground, time_shift_offset_placeholder_, [desired_samples])

        if is_training and background_data != []: #never enters, there's no background data
            background_volume_range = tf.constant(background_volume_range_,dtype=tf.float32)
            background_index = np.random.randint(len(background_data))
            background_samples = background_data[background_index]
            background_offset = np.random.randint(0, len(background_samples) - desired_samples)
            background_clipped = background_samples[background_offset:(background_offset + desired_samples)]
            background_clipped = tf.squeeze(background_clipped)
            background_reshaped = tf.pad(background_clipped,[[0,desired_samples-tf.shape(wav_decoder)[-1]]])
            background_reshaped = tf.cast(background_reshaped, tf.float32)
            if np.random.uniform(0, 1) < background_frequency:
                background_volume = np.random.uniform(0, background_volume_range_)
            else:
                background_volume = 0
            background_volume_placeholder_ = tf.constant(background_volume,dtype=tf.float32)
            background_data_placeholder_ = background_reshaped
            background_mul = tf.multiply(background_data_placeholder_, background_volume_placeholder_)
            background_add = tf.add(background_mul, sliced_foreground)
            sliced_foreground = tf.clip_by_value(background_add, -1.0, 1.0)
        
        if model_settings['feature_type'] == 'mfcc':
            stfts = tf.signal.stft( sliced_foreground, 
                                    frame_length=model_settings['window_size_samples'], 
                                    frame_step=model_settings['window_stride_samples'], fft_length=None,
                                    window_fn=tf.signal.hann_window)
            
            spectrograms = tf.abs(stfts)
            num_spectrogram_bins = stfts.shape[-1]
            # default values used by contrib_audio.mfcc as shown here
            # https://kite.com/python/docs/tensorflow.contrib.slim.rev_block_lib.contrib_framework_ops.audio_ops.mfcc
            lower_edge_hertz, upper_edge_hertz, num_mel_bins = 20.0, 4000.0, 40 
            linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, 
                                                                                num_spectrogram_bins,
                                                                                model_settings['sample_rate'],
                                                                                lower_edge_hertz, upper_edge_hertz)
            
            mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
            mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
            # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
            log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
            # Compute MFCCs from log_mel_spectrograms and take the first 13.
            mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :model_settings['dct_coefficient_count']]
            mfccs = tf.reshape(mfccs,[model_settings['spectrogram_length'], model_settings['dct_coefficient_count'], 1])
            next_element['audio'] = mfccs
        else:
            #in the original function we had: "elif feature_type==lfbe" and "elif feature_type==td_samples"
            raise Exception('Selected non-default feature type')
        
        return next_element

    return prepare_processing_graph
    
def prepare_background_data(bg_path,BACKGROUND_NOISE_DIR_NAME):
    """Searches a folder for background noise audio, and loads it into memory.
    It's expected that the background audio samples will be in a subdirectory
    named '_background_noise_' inside the 'data_dir' folder, as .wavs that match
    the sample rate of the training data, but can be much longer in duration.
    If the '_background_noise_' folder doesn't exist at all, this isn't an
    error, it's just taken to mean that no background noise augmentation should
    be used. If the folder does exist, but it's empty, that's treated as an
    error.
    Returns:
        List of raw PCM-encoded audio samples of background noise.
    Raises:
        Exception: If files aren't found in the folder.
    """
    background_data = []
    background_dir = os.path.join(bg_path, BACKGROUND_NOISE_DIR_NAME)
    if not os.path.exists(background_dir):
        return background_data

    search_path = os.path.join(bg_path, BACKGROUND_NOISE_DIR_NAME,'*.wav')

    for wav_path in gfile.Glob(search_path):
        raw_audio = tf.io.read_file(wav_path)
        audio = tf.audio.decode_wav(raw_audio)
        background_data.append(audio[0])
    if not background_data:
        raise Exception('No background wav files were found in ' + search_path)
    return background_data

def get_training_data(Flags,dataset_dir,background_dir, quarter_dataset ,val_cal_subset=False, perf=False):   #(Flags,get_waves=False,val_cal_subsets=False)
    label_count=12
    background_frequency = Flags['background_frequency']
    background_volume_range_= Flags['background_volume']
    model_settings = prepare_model_settings(label_count, Flags)

    bg_path=background_dir
    BACKGROUND_NOISE_DIR_NAME='_background_noise_' 
    background_data = prepare_background_data(bg_path,BACKGROUND_NOISE_DIR_NAME)

    splits = ['train', 'test', 'validation']
    
    if quarter_dataset == 1:
        (ds_train, ds_test, ds_val)= tfds.load('speech_commands', split = ['train[:25%]','test','validation[:25%]'],data_dir=dataset_dir) #QUARTER DATASET
    else:
        (ds_train, ds_test, ds_val)= tfds.load('speech_commands', split = splits,data_dir=dataset_dir)                                     #FULL DATASET
    
    if val_cal_subset:  # only return the subset of val set used for quantization calibration
        with open("quant_cal_idxs.txt") as fpi:
            cal_indices = [int(line) for line in fpi]
        cal_indices.sort()
        # cal_indices are the positions of specific inputs that are selected to calibrate the quantization
        count = 0  # count will be the index into the validation set.
        val_sub_audio = []
        val_sub_labels = []
        for d in ds_val:
            if count in cal_indices:          # this is one of the calibration inpus
                new_audio = d['audio'].numpy()  # so add it to a stack of tensors 
                if len(new_audio) < 16000:      # from_tensor_slices doesn't work for ragged tensors, so pad to 16k
                    new_audio = np.pad(new_audio, (0, 16000-len(new_audio)), 'constant')
                val_sub_audio.append(new_audio)
                val_sub_labels.append(d['label'].numpy())
            count += 1
        # and create a new dataset for just the calibration inputs.
        ds_val = tf.data.Dataset.from_tensor_slices({"audio": val_sub_audio,
                                                "label": val_sub_labels})
    
    if Flags['num_train_samples'] != -1:
        ds_train = ds_train.take(Flags['num_train_samples'])
    if Flags['num_val_samples'] != -1:
        ds_val = ds_val.take(Flags['num_val_samples'])
    if Flags['num_test_samples'] != -1:
        ds_test = ds_test.take(Flags['num_test_samples'])

    #if get_waves removed because def=false

    # extract spectral features and add background noise
    ds_train = ds_train.map(get_preprocess_audio_func(  model_settings,
                                                        is_training=True,
                                                        background_data=background_data),
                                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    ds_test  =  ds_test.map(get_preprocess_audio_func(  model_settings,
                                                        is_training=False,
                                                        background_data=background_data),
                                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    ds_val   =   ds_val.map(get_preprocess_audio_func(  model_settings,
                                                        is_training=False,
                                                        background_data=background_data),
                                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # change output from a dictionary to a feature,label tuple
    ds_train = ds_train.map(convert_dataset)
    ds_test = ds_test.map(convert_dataset)
    ds_val = ds_val.map(convert_dataset)

    # Now that we've acquired the preprocessed data, either by processing or loading,
    ds_train = ds_train.batch(Flags['batch_size'])
    if perf == 0:
        ds_test = ds_test.batch(Flags['batch_size'])  
    else:
        ds_test = ds_test.batch(1)
    ds_val = ds_val.batch(Flags['batch_size'])

    return ds_train, ds_test, ds_val
