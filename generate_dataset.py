import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
# Initialize Tensorflow 
tf.keras.layers.Dense(100)
from src.utils.gpu_memory_grow import gpu_memory_grow

from src.style_time import style_time
from src.data import data_loader
from src.utils.utils import extract_labels
import multiprocessing
import time
import shutil

from itertools import product

gpus = tf.config.list_physical_devices('GPU')
gpu_memory_grow(gpus)

GENERATED_SEQUENCES_FOLDER = "data/generated"
CONTENT_DATA_PATH = "data/simulated_dataset/01 - Source Domain.h5"

STYLE_DATASETS = [
        "data/simulated_dataset/amplitude_shift/1.0_1.0.h5", 
        "data/simulated_dataset/amplitude_shift/2.0_2.0.h5", 
        "data/simulated_dataset/amplitude_shift/3.0_3.0.h5", 
        "data/simulated_dataset/amplitude_shift/4.0_4.0.h5", 
        "data/simulated_dataset/amplitude_shift/5.0_5.0.h5", 
        "data/simulated_dataset/amplitude_shift/6.0_6.0.h5", 
        "data/simulated_dataset/amplitude_shift/7.0_7.0.h5" , 
        "data/simulated_dataset/amplitude_shift/8.0_8.0.h5" , 
        "data/simulated_dataset/amplitude_shift/9.0_9.0.h5" , 
        "data/simulated_dataset/amplitude_shift/10.0_10.0.h5",

        "data/simulated_dataset/output_noise/0.25.h5",
        "data/simulated_dataset/output_noise/0.50.h5",
        "data/simulated_dataset/output_noise/0.75.h5",
        "data/simulated_dataset/output_noise/1.00.h5",
        "data/simulated_dataset/output_noise/1.25.h5",
        "data/simulated_dataset/output_noise/1.50.h5",
        "data/simulated_dataset/output_noise/1.75.h5",
        "data/simulated_dataset/output_noise/2.00.h5",
        "data/simulated_dataset/output_noise/2.25.h5",
        "data/simulated_dataset/output_noise/2.50.h5",

        "data/simulated_dataset/time_shift/0.h5",
        "data/simulated_dataset/time_shift/2.h5",
        "data/simulated_dataset/time_shift/4.h5",
        "data/simulated_dataset/time_shift/6.h5",
        "data/simulated_dataset/time_shift/8.h5",
        "data/simulated_dataset/time_shift/10.h5",
        "data/simulated_dataset/time_shift/12.h5",
        "data/simulated_dataset/time_shift/14.h5",
        "data/simulated_dataset/time_shift/16.h5",
        "data/simulated_dataset/time_shift/18.h5"
    ]

def get_name(path:str):
    filename = path.split("/")[-1]
    return ".".join(filename.split('.')[:-1])




def generate(
    dset_content:tf.data.Dataset, 
    dset_style:tf.data.Dataset,
    progress_dict, 
    return_dict, 
    process_id, 
    n_sequences=2500):
    
    sequences, labels = [], []
    content_style_dataset = tf.data.Dataset.zip(dset_content, dset_style)
    
    for i, ((ts_content, label), (ts_style, _)) in enumerate(content_style_dataset.take(n_sequences)):
        generated_sequence, _,_,_ = style_time(ts_content, ts_style, iterations=1500)
        
        progress_dict[process_id] = (i, n_sequences)

        sequences.append(generated_sequence)
        labels.append(label)
    
    sequences = np.array(sequences)
    labels = np.array(labels)
    
    return_dict[process_id] = (sequences, labels)
    

def print_progress(_i, _n_sequences, terminal_width):
    percentage = _i/_n_sequences
    
    placeholder = f"{_i}/{_n_sequences}: {percentage*100:0.3f}%"
    tw = terminal_width - len(placeholder) -10
    
    bar = "#"* int(percentage*tw) + " "* int((1- percentage)* tw)
    print(f"{placeholder} [{bar}]")

def make_generated_dataset(
        content_path:str, 
        style_path:str, 
        progress_dict,
        result_dict,
        style_id):

    dset_content_train, dset_content_valid = data_loader.loading_wrapper(content_path, 64, 1, 0.5, 1, shuffle=True, drop_labels=False)
    dset_style_train, dset_style_valid = data_loader.loading_wrapper(style_path, 64, 1, 0.5, 1, shuffle=True, drop_labels=False)
    
    # Extract labels.
    labeled_content_train, labeled_content_valid  = extract_labels(dset_content_train), extract_labels(dset_content_valid)
    labeled_style_train, labeled_style_valid = extract_labels(dset_style_train), extract_labels(dset_style_valid)
    
    p1 = multiprocessing.Process(target=generate, args=(labeled_content_train, labeled_style_train, progress_dict, result_dict, style_id+'_train'))
    p2 = multiprocessing.Process(target=generate, args=(labeled_content_valid, labeled_style_valid, progress_dict, result_dict, style_id+'_valid', 500))
    
    p1.start()
    p2.start()
    
    p1.join()
    p2.join()

if __name__ == "__main__":
        
    manager = multiprocessing.Manager()
    progress_dict = manager.dict()  # Shared dictionary to track progress
    results_dict = manager.dict()
    
    processes = []
    for i, style_path in enumerate(STYLE_DATASETS):
        style_name = get_name(style_path)
        progress_dict[style_name+"_train"] = (0, 10) # Initialize progress for each process
        progress_dict[style_name+"_valid"] = (0, 10) # Initialize progress for each process
        
        results_dict[style_name+"_train"]= None
        results_dict[style_name+"_valid"]= None
        p = multiprocessing.Process(target=make_generated_dataset, args=(CONTENT_DATA_PATH, style_path, progress_dict, results_dict, style_name))
        processes.append(p)
        p.start()

    while any(p.is_alive() for p in processes):
        terminal_size = shutil.get_terminal_size()
        # Width of the terminal
        terminal_width = terminal_size.columns
        os.system('cls' if os.name == 'nt' else 'clear')

        print('[+] TRAIN SET [+]')
        # Print progress for each task on its own line
        for style_path in STYLE_DATASETS:
            key = get_name(style_path)+"_train"
            _i, _n_sequences = progress_dict[key]
            
            print_progress(_i, _n_sequences, terminal_width)
            
        print('[+] VALID SET [+]')
        # Print progress for each task on its own line
        for style_path in STYLE_DATASETS:
            key = get_name(style_path)+"_valid"
            _i, _n_sequences = progress_dict[key]
            
            print_progress(_i, _n_sequences, terminal_width)
            
        time.sleep(1.)  # Refresh rate
        # print("####")

    for p in processes:
        p.join()
    
    for k in results_dict.keys():
        data_sequences, data_labels = results_dict[k]
        data_sequences = tf.data.Dataset.from_tensor_slices(data_sequences)
        data_labels = tf.data.Dataset.from_tensor_slices(data_labels)
        
        dset = tf.data.Dataset.zip(data_sequences, data_labels)
        dset.save(f"{GENERATED_SEQUENCES_FOLDER}/{k}.tfrecords")
        
