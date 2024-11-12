import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from src.data import data_loader as dataLoader

def extract_labels(dset) -> tf.data.Dataset:
    idx = 32
    return dset.map(lambda seq: (seq[:, :, :-1], seq[:, idx, -1]))

def get_name(path:str):
    filename = path.split("/")[-1]
    return ".".join(filename.split('.')[:-1])


def extract_labels(dset) -> tf.data.Dataset:
    idx = 32

    return dset.map(lambda seq: (seq[:, :, :-1], seq[:, idx, -1]))


def load_dset(df_path:str, drop_labels=False, bs = 64) -> tf.data.Dataset:
    sequence_length = 64
    gran = 1
    overlap = 0.05
    
    return dataLoader.loading_wrapper(df_path, sequence_length, gran, overlap, bs, drop_labels=drop_labels)


def plot_several_sequence(dset:np.ndarray, count:int):
    """Take a (n_sequences, n_features, sequence_length) and display `count` sequences

    Args:
        dset (np.ndarray): A np array of shape (n_sequences, n_features, sequence_length);
        count (int): The number of sequences to display.;
    """
    for i in range(count):
        fig = plt.figure(figsize=(18, 3))
        
        plt.plot(dset[i][0], '.-')
        plt.grid()
        plt.show()
        plt.close(fig)