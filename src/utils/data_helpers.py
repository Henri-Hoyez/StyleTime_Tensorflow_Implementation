import numpy as np 
import pandas as pd


def train_test_split(dset:np.ndarray, train_size=0.7):
    shape = dset.shape
    train_idx = int(shape[0]*train_size)
    
    dset = dset.reshape((shape[0], 1, shape[-1])).astype(np.float32)
    
    train = dset[:train_idx]
    test = dset[train_idx:]
    
    return train, test 


def make_dataset(dset:np.ndarray):
    rng = np.random.default_rng()
    window_length = 30
    sequences = []
    
    for i in range(0, dset.shape[0]- window_length):
        sequence = dset[i: i+window_length]
        sequences.append(sequence)
        
    return rng.permutation(np.array(sequences))


def normalize(dataframe:pd.DataFrame):
    _min, _max = dataframe.min(), dataframe.max()

    return (dataframe - _min)/(_max - _min)


def standardize(dataframe:pd.DataFrame):
    _mean, _std = dataframe.mean(), dataframe.std()

    return (dataframe - _mean)/(_std)

####

def make_step_function(
    x:np.ndarray, 
    step_time:float, 
    step_value:float):
    
    _step = np.zeros_like(x)
    _step[:, step_time:] = step_value    
    
    return x + _step


def make_perturbed_dataset(dset:pd.DataFrame):
    rng = np.random.default_rng() # Applying the new numpy Generator.
    perturbed_dataset = []
    choices = [-1, 1]

    for sequence in dset:
        random_time = int(np.random.uniform(0, sequence.shape[1]))
        std_value= np.std(sequence)* 5
        
        rnd_idx = np.random.choice(2)
        random_value= std_value* choices[rnd_idx]
                
        perturbed_dataset.append(make_step_function(sequence, random_time, random_value))
    
    return np.array(perturbed_dataset)
