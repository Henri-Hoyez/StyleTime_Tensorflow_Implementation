import numpy as np

import os

from utils import gpu_memory_grow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # This will make TensorFlow run on CPU

import tensorflow as tf
from tensorflow import keras
from src.models import losses
from src.data import data_loader
import matplotlib.pyplot as plt


# This is for force the initialization of keras.
test = keras.layers.Dense(100)
gpus = tf.config.list_physical_devices('GPU')
gpu_memory_grow.gpu_memory_grow(gpus)


# StyleTime Algorithm
def style_time(content_series, style_series, iterations=5000, alpha=0.5 , beta=1000.0, gamma=0.0001, learning_rate=0.01):
    """Implements the StyleTime algorithm."""
    
    # Convert the input series to tensors
    y = tf.Variable(content_series, dtype=tf.float32)
    
    c_losses = []
    s_losses = []
    total_losses = []
    
    
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    
    for i in range(iterations):
        with tf.GradientTape() as tape:
            # Compute total loss
            c_loss = losses.content_loss(y, content_series)
            s_loss = losses.style_loss(y, style_series)
            tv_loss = losses.total_variation_loss(y)
            
            total_loss = alpha * c_loss + beta * s_loss + gamma * tv_loss
            
        c_losses.append(c_loss.numpy())
        s_losses.append(s_loss.numpy())
        total_losses.append(total_loss.numpy())
            
        # Compute gradients and apply optimization
        gradients = tape.gradient(total_loss, [y])
        # print(gradients)
        optimizer.apply_gradients(zip(gradients, [y]))
        
        if i % 10 == 0:
            print(f"\rIteration {i}/{iterations}. content loss {alpha* c_loss:0.2f}, Style loss {beta* s_loss:0.2f}, tv loss {gamma * tv_loss:0.2f}, Total Loss: {total_loss.numpy():0.2f}", end="")
        # exit()

    return y.numpy(), c_losses, s_losses, total_losses



# Example Usage
if __name__ == '__main__':
    
    content_data_path = "data/simulated_dataset/01 - Source Domain.h5"
    style_data_path = "data/simulated_dataset/amplitude_shift/2.0_2.0.h5"
    
    dset_content_train, dset_content_valid = data_loader.loading_wrapper(content_data_path, 64, 1, 0.5, 1, shuffle=True)
    dset_style_train, dset_style_valid = data_loader.loading_wrapper(style_data_path, 64, 1, 0.5, 1, shuffle=True)

    
    # Sample content and style time series
    content_series = next(iter(dset_content_train))
    style_series = next(iter(dset_style_train))

    # Run StyleTime algorithm
    synthetic_series, c_losses, s_losses, total_losses = style_time(content_series, style_series)
    
    # Print the generated series
    # print("Generated Synthetic Series:", synthetic_series)
    
    # plot figure
    plt.figure(figsize=(18, 10))
    ax = plt.subplot(311)
    ax.set_title("Content TS")
    ax.plot(content_series[0])
    ax.set_ylim(0, 20)
    ax.grid()
    
    ax = plt.subplot(312)
    ax.set_title("Style TS")
    ax.plot(style_series[0])
    ax.set_ylim(0, 20)
    ax.grid()
    
    ax = plt.subplot(313)
    ax.set_title("Generated TS")
    ax.plot(synthetic_series[0])
    ax.set_ylim(0, 20)
    ax.grid()
    
    plt.savefig("Sequence.png")
    
    
    plt.figure(figsize=(18, 10))
    ax = plt.subplot(311)
    ax.set_title("Content TS")
    ax.plot(c_losses, label="content losses.")
    ax.grid()
    
    
    ax = plt.subplot(312)
    ax.set_title("Style Loss")
    ax.plot(s_losses, label="style losses.")
    ax.grid()
    
    
    ax = plt.subplot(313)
    ax.set_title("Total Loss")
    ax.plot(total_losses, label="total losses.")

    ax.grid()
    
    plt.legend()

    
    plt.savefig("losses.png")
    
    
    
    
    
    
    
    


