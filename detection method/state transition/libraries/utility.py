import numpy as np
import pandas as pd

def load_sample(file_path):
        data = np.load(file_path, allow_pickle=False)
        return data

def write_to_csv(data, name):
    '''
    data in dict format, where keys form the column names
    '''
    df = pd.DataFrame(data)
    df.to_csv(name+'.csv', index=False)