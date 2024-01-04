import numpy as np
import pandas as pd
import json

def load_sample(file_path):
        '''
        for numpy files
        '''
        data = np.load(file_path, allow_pickle=False)
        return data


def read_traces(log_path):
    '''
    read the trace files and extract variable names
    data = [ [event, timestamp], [], [],......,[] ]
    '''
    with open(log_path, 'r') as f:
        data = json.load(f)
    return data


def write_to_csv(data, name):
    '''
    data in dict format, where keys form the column names
    '''
    df = pd.DataFrame(data)
    df.to_csv(name+'.csv', index=False)