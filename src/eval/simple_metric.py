
import numpy as np
from src.eval import metric

def moving_average(arr, n=3):
    return np.array([np.convolve(arr[:, i], np.ones(n)/n, mode='same') for i in range(arr.shape[-1])]).T

def simple_metric_on_noise(signals:np.ndarray):
    # Here the goal is, knowing the noise domain shift, 
    # extract the trend and the noise from a the signal.
    trends = np.array([moving_average(s) for s in signals])

    residual_noises = np.mean(np.std(trends - signals, axis=1), axis=0)
    return trends, residual_noises

def estimate_time_shift(batch, anchor_senssor:int, compared_sensssor:int):
    anchor_sequence= np.diff(batch[:, :-32, anchor_senssor])
    compared_sequence = batch[:, :, compared_sensssor]

    correlations = metric.optimized_windowed_cov(anchor_sequence, compared_sequence, beta=0).T

    correlations = np.mean(correlations, axis=0)
    
    t_max = np.argmax(correlations)

    return t_max

def extract_amplitude_from_signals(batch):
    _mins = np.mean(np.min(batch, axis=1), axis=0)
    _maxs = np.mean(np.max(batch, axis=1), axis=0)

    return _maxs- _mins