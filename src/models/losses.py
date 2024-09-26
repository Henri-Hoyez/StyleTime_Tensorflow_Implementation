import numpy as np

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # This will make TensorFlow run on CPU


import tensorflow as tf

# Helper functions to extract stylized features from time series

def trend_extraction(series, window_size=5, axis=-1):
    """
    Extracts the trend from a time series or array of time series using a moving average.
    
    Args:
        series (np.array): A 1D, 2D, or 3D array where each time series is along the specified axis.
        window_size (int): The size of the moving average window.
        axis (int): The axis along which to apply the moving average (default is -1, the last axis).
        
    Returns:
        np.array: An array with the same shape as input, with the trend extracted for each time series.
    """
    # Ensure the input is a numpy array
    series = np.asarray(series)
    
    # Moving average kernel
    kernel = np.ones(window_size) / window_size
    
    # Apply the convolution along the specified axis
    if series.ndim == 1:  # 1D case
        return np.convolve(series, kernel, mode='same')
    else:
        # Handle multi-dimensional arrays
        return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis, series)
    

def autocorrelation(series, lag=1, axis=-1):
    """
    Computes the autocorrelation for a time series or array of time series along the specified axis.
    
    Args:
        series (np.array): A 1D, 2D, or 3D array where each time series is along the specified axis.
        lag (int): The lag at which to compute the autocorrelation (default is 1).
        axis (int): The axis along which to compute the autocorrelation (default is -1, the last axis).
        
    Returns:
        np.array: The autocorrelation values along the specified axis, with the same shape as input except the axis of interest.
    """
    # Ensure the input is a numpy array
    series = np.asarray(series)
    
    def single_autocorrelation(ts, lag):
        ts_mean = np.mean(ts)
        numerator = np.sum((ts[:-lag] - ts_mean) * (ts[lag:] - ts_mean))
        denominator = np.sum((ts - ts_mean) ** 2)
        return numerator / denominator if denominator != 0 else 0
    
    # Apply autocorrelation function along the specified axis
    if series.ndim == 1:
        # Single 1D array
        return single_autocorrelation(series, lag)
    else:
        # Multi-dimensional array, apply along the specified axis
        return np.apply_along_axis(lambda m: single_autocorrelation(m, lag), axis, series)



def volatility(series):
    """Returns the volatility (standard deviation of log returns) of the series."""
    log_returns = np.diff(np.log(series + 1e-8))
    return np.std(log_returns)

def power_spectral_density(series):
    """Computes the power spectral density (PSD) of the series."""
    fft = np.fft.fft(series)
    psd = np.abs(fft) ** 2
    return np.mean(psd)

# Loss Functions
def content_loss(y, y_c):
    """Computes the content loss (difference in trend)."""
    trend_yc = trend_extraction(y_c)
    return tf.reduce_sum(tf.square(y - trend_yc))

def style_loss(y, y_s):
    """Computes the style loss based on autocorrelation, volatility, and PSD."""
    autocorr_loss = tf.math.reduce_mean(tf.abs(autocorrelation(y) - autocorrelation(y_s)))
    volatility_loss = tf.abs(volatility(y) - volatility(y_s))
    psd_loss = tf.abs(power_spectral_density(y) - power_spectral_density(y_s))

    psd_loss = tf.cast(psd_loss, tf.float32)    
    
    return autocorr_loss + volatility_loss + psd_loss

def total_variation_loss(y):
    """Encourages smoothness in the generated series by penalizing large changes."""
    return tf.reduce_sum(tf.square(y[1:] - y[:-1]))