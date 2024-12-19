import numpy as np

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # This will make TensorFlow run on CPU

import tensorflow as tf
import pandas as pd

# Helper functions to extract stylized features from time series

def trend_extraction(series, window_size=2, axis=-1):
    """
    Extracts the trend from a time series or array of time series using a moving average.
    
    Args:
        series (np.array): A 1D, 2D, or 3D array where each time series is along the specified axis.
        window_size (int): The size of the moving average window.
        axis (int): The axis along which to apply the moving average (default is -1, the last axis).
        
    Returns:
        np.array: An array with the same shape as input, with the trend extracted for each time series.
    """
    # _series = pd.DataFrame(series.T)
    _series = pd.DataFrame(series[0])
    
    _means = _series.rolling(window_size, min_periods=0, center=True).mean().values
    
    return _means.reshape(series.shape)



def log_return(yt):
    return yt[:, 1:] - yt[:, :-1]  
    

def autocorrelation(series, lag=1, axis=1):
    """
    Computes the autocorrelation for a time series or array of time series along the specified axis.
    
    Args:
        series (np.array): A 1D, 2D, or 3D array where each time series is along the specified axis.
        lag (int): The lag at which to compute the autocorrelation (default is 1).
        axis (int): The axis along which to compute the autocorrelation (default is -1, the last axis).
        
    Returns:
        np.array: The autocorrelation values along the specified axis, with the same shape as input except the axis of interest.
    """

    ts_mean = tf.math.reduce_mean(log_return(series), axis=-1)

    numerator = tf.math.reduce_sum((series[:, :-lag] - ts_mean) * (series[:, lag:] - ts_mean))
    denominator = tf.math.reduce_sum((series - ts_mean) ** 2)

    return numerator / denominator if denominator != 0 else 0


def volatility(series, eps=1e-8):
    """Returns the volatility (standard deviation of log returns) of the series."""
    _mean = tf.math.reduce_mean(log_return(series), axis=1)
    
    _diff = tf.math.reduce_sum(tf.math.square(series - _mean))/(series.shape[1]-2)
    
    return tf.math.sqrt(_diff+ eps)


def power_spectral_density(series):
    """Computes the power spectral density (PSD) of the series."""
    fft = np.fft.fft(series)
    psd = np.abs(fft) ** 2
    return np.mean(psd)

# Loss Functions
def content_loss(y, y_c):
    """Computes the content loss (difference in trend)."""
    trend_yc = trend_extraction(y_c, axis=1)
    return tf.reduce_sum(tf.square(y - trend_yc))

def style_loss(y, y_s):
    """Computes the style loss based on autocorrelation, volatility, and PSD."""
    autocorr_loss = tf.math.reduce_mean(tf.abs(autocorrelation(y) - autocorrelation(y_s)))
    volatility_loss = tf.abs(volatility(y) - volatility(y_s))
    psd_loss = tf.abs(power_spectral_density(y) - power_spectral_density(y_s))


    volatility_loss = tf.cast(volatility_loss, tf.float32)  
    psd_loss = tf.cast(psd_loss, tf.float32)  
        
    return autocorr_loss + volatility_loss + psd_loss

def total_variation_loss(y):
    """Encourages smoothness in the generated series by penalizing large changes."""
    return tf.reduce_sum(tf.square(y[0, 1:] - y[0, :-1]))