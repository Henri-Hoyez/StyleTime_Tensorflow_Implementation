import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["TF_USE_LEGACY_KERAS"]="1"
import tensorflow as tf


def optimized_cov(a,v):
    nan_mean_a = np.nanmean(a, axis=1).reshape((-1,1))
    nan_mean_b = np.nanmean(v, axis=1).reshape((-1,1))
    return np.nansum((a- nan_mean_a)* (v- nan_mean_b), axis=1)

def mean_difference(a,v):
    return np.nanmean(a) - np.nanmean(v)

def optimized_windowed_cov(a, v, beta=0):
    if a.shape[1] > v.shape[1]:
        _a, _v = v, a 
    else: 
        _a, _v = a, v

    n = _a.shape[1]
    corrs = []

    aa = optimized_cov(_a, _a)
    for k in range(_v.shape[1] - _a.shape[1]):
        __v = _v[:, k: n+k]
        # Compute the covariance 

        av = optimized_cov(_a,__v)
        vv = optimized_cov(__v, __v)
        _mean_diff = beta* mean_difference(_a,__v)

        augmented_cov = av/np.sqrt(aa*vv)+ _mean_diff
        # /(np.sqrt(aa*vv)) 

        corrs.append(augmented_cov)
        
    return np.array(corrs)

def step_fonction(xs, treashold=0.5):
    return tf.cast(tf.math.greater_equal(xs, treashold), tf.float32)


def accuracy(y_true, y_pred):
    y_pred = step_fonction(y_pred)

    y_pred = tf.reshape(y_pred, (-1,))
    y_true = tf.reshape(y_true, (-1,))
    
    good_prediction = tf.reduce_sum(tf.cast(y_pred == y_true, tf.float32))

    return good_prediction/y_true.shape[0]