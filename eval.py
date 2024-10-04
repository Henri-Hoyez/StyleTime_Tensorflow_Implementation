import pandas as pd
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
tf.keras.layers.Dense(100)
from utils.gpu_memory_grow import gpu_memory_grow

import matplotlib.pyplot as plt
from models.evaluation import utils
from utils.utils import dataLoader, simple_metric



# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import umap
from sklearn.manifold import TSNE
import argparse
from utils.utils import visualization_helpersv2

gpus = tf.config.list_physical_devices('GPU')
gpu_memory_grow(gpus)


GENERATED_SEQUENCES_FOLDER = "data/generated"
CONTENT_DATA_PATH = "data/simulated_dataset/01 - Source Domain.h5"

SAVE_FOLDER = "results/amplitudes"

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
        "data/simulated_dataset/amplitude_shift/10.0_10.0.h5"
    ]



def get_name(path:str):
    filename = path.split("/")[-1]
    return ".".join(filename.split('.')[:-1])

def stylize(
        content_dset:tf.data.Dataset, 
        style_sequence:tf.Tensor, 
        content_encoder:tf.keras.Model, 
        style_encoder:tf.keras.Model, 
        decoder:tf.keras.Model) -> tf.data.Dataset:
    
    content_space = content_dset.map(lambda seq, _: (content_encoder(seq)), num_parallel_calls=tf.data.AUTOTUNE).cache()
    labels = content_dset.map(lambda _,l : l, num_parallel_calls=tf.data.AUTOTUNE).cache()
    
    style_vector = style_encoder(style_sequence)

    translated = content_space.map(lambda c: tf.concat(decoder([c, style_vector], training=False), -1), num_parallel_calls=tf.data.AUTOTUNE).cache()
    dset_final = tf.data.Dataset.zip((translated, labels))

    return dset_final

def get_generated_dataset(real_style_path:list, gen_folder:str=GENERATED_SEQUENCES_FOLDER):
    filename = get_name(real_style_path)
    
    train_gen_filepath = f"{gen_folder}/{filename}_train.tfrecords"
    valid_gen_filepath = f"{gen_folder}/{filename}_valid.tfrecords"
    
    dset_train = tf.data.Dataset.load(train_gen_filepath)
    dset_valid = tf.data.Dataset.load(valid_gen_filepath)
    
    return dset_train, dset_valid

def get_real_fake_datasets(style_datasets):
    real_style_dataset = {}
    fake_style_dataset = {}
    bs = 256 #args().simulated_arguments.batch_size

    for style_path in style_datasets:
        sty_name = get_name(style_path)

        dset_style_train, dset_style_valid = utils.load_dset(style_path, drop_labels=False, bs=bs)

        stylized_train, stylized_valid = get_generated_dataset(style_path, GENERATED_SEQUENCES_FOLDER)

        fake_style_dataset[f"{sty_name}_train"] = stylized_train
        fake_style_dataset[f"{sty_name}_valid"] = stylized_valid
        
        real_style_dataset[f"{sty_name}_train"] = dset_style_train
        real_style_dataset[f"{sty_name}_valid"] = dset_style_valid
        
    return real_style_dataset, fake_style_dataset

def tstr(
    dset_train_real,
    dset_valid_real,
    dset_train_fake, 
    dset_valid_fake, 
    save_to:str):

    print('[+] Train Real, Test Real.')
    real_performances, hist_real = eval_methods.train_naive_discriminator(dset_train_real, dset_valid_real, args(), epochs=5, n_classes=5)

    print("[+] Train Synthetic, Test Synthetic")
    gen_perf1, hist_fake1 = eval_methods.train_naive_discriminator(dset_train_fake, dset_valid_fake, args(), epochs=5, n_classes=5)
    
    print("[+] Train Synthetic, Test Real")
    gen_perf2, hist_fake2 = eval_methods.train_naive_discriminator(dset_train_fake, dset_valid_real, args(), epochs=5, n_classes=5)
    
    
    fig = plt.figure(figsize=(18, 10))
    
    ax = plt.subplot(211)
    
    plt.plot(hist_real.history["loss"], ".-", label='Train Real Test Real (Train)')
    plt.plot(hist_real.history["val_loss"], ".-", label='Train Real Test Real (Valid)')
    
    plt.plot(hist_fake1.history["loss"], ".-", label='Train Synthetic, Test Synthetic (Train)')
    plt.plot(hist_fake1.history["val_loss"], ".-", label='Train Synthetic, Test Synthetic (Valid)')
    
    plt.plot(hist_fake2.history["loss"], ".-", label='Train Real, Test Synthetic (Train)')
    plt.plot(hist_fake2.history["val_loss"], ".-", label='Train Real, Test Synthetic (Valid)')
    
    ax.legend()
    ax.grid()
    ax = plt.subplot(212)
    
    plt.plot(hist_real.history["sparse_categorical_accuracy"], ".-", label='Classification Acc on Real (Train)')
    plt.plot(hist_real.history["val_sparse_categorical_accuracy"], ".-", label='Classification Acc on Real (Valid)')
    
    plt.plot(hist_fake1.history["sparse_categorical_accuracy"], ".-", label='Train Synthetic, Test Synthetic (Train)')
    plt.plot(hist_fake1.history["val_sparse_categorical_accuracy"], ".-", label='Train Synthetic, Test Synthetic (Valid)')
    
    plt.plot(hist_fake2.history["sparse_categorical_accuracy"], ".-", label='Train Real, Test Synthetic (Train)')
    plt.plot(hist_fake2.history["val_sparse_categorical_accuracy"], ".-", label='Train Real, Test Synthetic (Valid)')
    
    ax.grid()
    ax.legend()
    
    plt.savefig(save_to)
    
    plt.close(fig)
    
    return real_performances, gen_perf2


def tstr_on_styles(real_dataset, fake_dataset, style_names, model_folder):
    tstr_stats = {}

    for i, style_ in enumerate(style_names):
        print(f'[+] Training on dataset {style_}.')
        
        perf_on_real, perf_on_fake = tstr(
            real_dataset[f"{style_}_train"],
            real_dataset[f"{style_}_valid"],
            fake_dataset[f"{style_}_train"],
            fake_dataset[f"{style_}_valid"], 
            f'{model_folder}/tstr_{style_}.png'
            )
        
        tstr_stats[f"{style_}_real"] = [perf_on_real]
        tstr_stats[f"{style_}_gen"] = [perf_on_fake]
        
    tstr_stats = pd.DataFrame.from_dict(tstr_stats)

    tstr_stats.to_hdf(f"{model_folder}/tstr.h5", key="data")
    
    return tstr_stats

def plot_tstr_results(tstr_stats:pd.DataFrame, model_folder:str):
    def remove_prefix(cols:list):
        return [c.split("_")[0] for c in cols]
    
    tstr_real = tstr_stats.filter(like='real', axis=1)
    tstr_fake = tstr_stats.filter(like='gen', axis=1)

    tstr_real.columns = remove_prefix(tstr_real.columns)
    tstr_fake.columns = remove_prefix(tstr_fake.columns)
    
    plt.figure(figsize=(18, 10))
    ax = plt.subplot(111)
    ax.set_title("Gap Between Accuracies.")

    plt.plot(tstr_real.values.reshape((-1,)), ".-", label='Acc on Real, trained on Real')
    plt.plot(tstr_fake.values.reshape((-1,)), ".-", label='Acc on Real, trained on Fake')

    ax.set_xticklabels(tstr_real.columns.values)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Amplitudes")

    ax.grid(True)
    ax.legend()
    
    plt.savefig(f"{model_folder}/tstr.png")
    
    
def get_batches(dset, n_batches):
    _arr = np.array([c[0] for c in dset.take(n_batches)])
    return _arr.reshape((-1, _arr.shape[-2], _arr.shape[-1]))
    
    
def compute_metrics(dset_real, dset_fake, style_names):
    def time_shift_evaluation(big_batch):
        return [simple_metric.estimate_time_shift(big_batch, 0, i) for i in range(big_batch.shape[-1])]
    
    real_noise_metric, gen_noise_metric = [], []
    real_ampl_metric, gen_ampl_metric = [], []
    real_ts_metric, gen_ts_metric = [], []

    for style_name in style_names:
        print(f"[+] Compute metric for {style_name}")
        real_batch = get_batches(dset_real[f"{style_name}_valid"], 10)
        fake_batch = get_batches(dset_fake[f"{style_name}_valid"], 10)
        
        real_noise_metric.append(simple_metric.simple_metric_on_noise(real_batch)[-1])
        gen_noise_metric.append(simple_metric.simple_metric_on_noise(fake_batch)[-1])
        
        real_ampl_metric.append(simple_metric.extract_amplitude_from_signals(real_batch))
        gen_ampl_metric.append(simple_metric.extract_amplitude_from_signals(fake_batch))
        
        real_ts_metric.append(time_shift_evaluation(real_batch))
        gen_ts_metric.append(time_shift_evaluation(fake_batch))
        
    real_mean_noises = np.mean(real_noise_metric, axis=-1).reshape((-1, 1))
    fake_mean_noises = np.mean(gen_noise_metric, axis=-1).reshape((-1, 1))
    mean_noises = np.concatenate((real_mean_noises, fake_mean_noises), axis=-1)
    
    real_mean_ampl = np.mean(real_ampl_metric, axis=-1).reshape((-1, 1))
    fake_mean_ampl = np.mean(gen_ampl_metric, axis=-1).reshape((-1, 1))
    mean_ampl= np.concatenate((real_mean_ampl, fake_mean_ampl), axis=-1)
    
    real_mean_time_shift = np.mean(real_ts_metric, axis=-1).reshape((-1, 1))
    fake_mean_time_shift = np.mean(gen_ts_metric, axis=-1).reshape((-1, 1))
    mean_time_shift= np.concatenate((real_mean_time_shift, fake_mean_time_shift), axis=-1)
    
    df_noises = pd.DataFrame(data=mean_noises, index=style_names, columns=['Real', 'Fake'])
    df_ampl = pd.DataFrame(data=mean_ampl, index=style_names, columns=['Real', 'Fake'])
    df_time_shift = pd.DataFrame(data=mean_time_shift, index=style_names, columns=['Real', 'Fake'])
    
    return df_noises, df_ampl, df_time_shift


def plot_metric(df_metric:pd.DataFrame, title, save_to):
    plt.figure(figsize=(18, 10))
    ax = plt.subplot(111)
    
    df_metric["Real"].plot(ax=ax, style='.-')
    df_metric["Fake"].plot(ax=ax, style='.-')
    
    ax.grid(True)
    ax.set_title(title)
    ax.legend()
    
    plt.savefig(save_to)
    
def multi_umap_plot(real_styles, gen_styles):
    (_, _, seq_len, n_sigs) = real_styles.shape
    
    concatenated = tf.concat((real_styles, gen_styles), 0)

    concatenated = tf.reshape(concatenated, (-1, seq_len, n_sigs))
    concatenated = tf.transpose(concatenated, (0, 2, 1))
    
    concatenated = tf.reshape(concatenated, (concatenated.shape[0], -1))

    # # # Normalize all sequences for the reducer.
    _mean, _std = tf.math.reduce_mean(concatenated), tf.math.reduce_std(concatenated)
    concatenated = (concatenated - _mean)/_std

    reducer = umap.UMAP(n_neighbors=300, min_dist=1., random_state=42, metric="euclidean") 
    reduced = reducer.fit_transform(concatenated)
    return reduced


def multi_tsne_plot(real_styles, gen_styles):
    (_, _, seq_len, n_sigs) = real_styles.shape
    
    concatenated = tf.concat((real_styles, gen_styles), 0)

    concatenated = tf.reshape(concatenated, (-1, seq_len, n_sigs))
    concatenated = tf.transpose(concatenated, (0, 2, 1))
    
    concatenated = tf.reshape(concatenated, (concatenated.shape[0], -1))

    # # # Normalize all sequences for the reducer.
    _mean, _std = tf.math.reduce_mean(concatenated), tf.math.reduce_std(concatenated)
    concatenated = (concatenated - _mean)/_std

    reducer = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=150, random_state=42)
    reduced = reducer.fit_transform(concatenated)
    return reduced


def generate_per_style_batch(dset_real, dset_fake, style_names):
    real_batches = []
    fake_batches = []

    for _, style_ in enumerate(style_names):
        real_style_batch = get_batches(dset_real[f"{style_}_valid"], 10)
        fake_style_batch = get_batches(dset_fake[f"{style_}_valid"], 10)
        
        real_batches.append(real_style_batch)
        fake_batches.append(fake_style_batch)
        
    return np.array(real_batches), np.array(fake_batches) 


def dimentionality_reduction_plot(real_batches, fake_batches, style_names, model_folder, type="umap"):
    if type == 'umap':
        reduced_points = multi_umap_plot(real_batches, fake_batches)
    elif type == "tsne":
        reduced_points = multi_tsne_plot(real_batches, fake_batches)
    else: 
        raise Exception("No Dimentionality reduction algorthm selected.")
        
    
    n_styles = len(style_names)

    (n_styles, bs, seq_len, n_sigs) = real_batches.shape

    cmap = plt.get_cmap("tab20")
    colors = cmap(np.linspace(0, 1, n_styles*2))

    plt.figure(figsize=(18, 10))
    for i in range(n_styles):
        ri, ro = i*bs, (i+1)*bs
        fi, fo =  (i+ n_styles) * bs, (i+ n_styles+ 1) * bs
        
        plt.scatter(reduced_points[ri:ro, 0], reduced_points[ri:ro, 1], label=f"Real Style {i+ 1}", alpha=0.5, color=colors[2*i], s=4)
        plt.scatter(reduced_points[fi:fo, 0], reduced_points[fi:fo, 1], label=f"Generated Style {i+ 1}", alpha=0.5, color=colors[2*i+1 ], s=4)
    plt.grid()
    plt.title(f"{type} Reduction of Time Series", fontsize=15)
    plt.ylabel(f"y_{type}", fontsize=15)
    plt.xlabel(f"x_{type}", fontsize=15)
    plt.legend()
    plt.savefig(f"{model_folder}/{type}.png")
    plt.show()
    




def main():
        
    style_names = [get_name(p) for p in STYLE_DATASETS]
    
    dsets_real, dsets_fake = get_real_fake_datasets(STYLE_DATASETS)
    
    exit()
    
    tstr_stats = tstr_on_styles(dsets_real, dsets_fake, style_names, shell_arguments.model_folder)
    
    df_noises, df_ampl, df_time_shift = compute_metrics(dsets_real, dsets_fake, style_names)
    
    plot_metric(df_ampl, "Amplitude metric comparison", f"{shell_arguments.model_folder}/amplitude_metric_comparison.png")
    plot_metric(df_noises, "Noise metric comparison", f"{shell_arguments.model_folder}/noise_metric_comparison.png")
    plot_metric(df_time_shift, "time shift metric comparison", f"{shell_arguments.model_folder}/time shift comparison.png")
    
    real_batches, fake_batches = generate_per_style_batch(dsets_real, dsets_fake, style_names)
    
    dimentionality_reduction_plot(real_batches, fake_batches, style_names, shell_arguments.model_folder, "umap")
    dimentionality_reduction_plot(real_batches, fake_batches, style_names, shell_arguments.model_folder, "tsne")
    
    





if __name__ == '__main__':
    main()
