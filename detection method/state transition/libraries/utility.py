import numpy as np
import pandas as pd
import json
import os

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


def get_paths(log_path: str):
    '''
    log_path: path to the folder containing the log files -> str

    return: 
    paths to the log files -> list;
    traces files -> list;
    varlist files -> list;
    '''
    label_path = os.path.join(log_path, 'labels')
    if os.path.exists(label_path):
        labels = os.listdir(label_path)
    else:
        labels = []
    all_files = os.listdir(log_path)
    all_files.sort()
    logs = []
    traces = []
    varlist = []
    unknown = []
    for i in all_files:
        # if i.find('label') == 0:
        #     labels += [i]
        if i.find('log') == 0:
            logs += [i]
        elif i.find('trace') == 0 and i.find('.txt') == -1:
            traces += [i]
        elif i.find('varlist') == 0:
            varlist += [i]
        else:
            unknown += [i]

    ######### path to files
    paths_log = [os.path.join(log_path, x) for x in logs]
    paths_traces = [os.path.join(log_path, x) for x in traces]
    varlist_path = [os.path.join(log_path,x) for x in varlist]
    paths_label = [os.path.join(label_path, x) for x in labels]
    paths_log.sort()
    paths_traces.sort()
    varlist_path.sort()
    paths_label.sort()

    return paths_log, paths_traces, varlist_path, paths_label
