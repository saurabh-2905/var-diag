from collections import defaultdict
import numpy as np
import os
from libraries.utility import load_sample, write_to_csv
import json


class StateTransition:

    def __init__(self):
        self.transitions = defaultdict(list)


    def train(self, file_paths):
        '''
        file_paths -> list: 
            complete path to the sample data files (.npy)
        '''
        for sample_path in file_paths:
            sample_data = load_sample(sample_path)
            print(sample_path)

            for event1, event2 in zip(sample_data[0:-1], sample_data[1:]):
                # print(event1,event2)
                var1, var2 = event1[0], event2[0]
                # print(var1, var2)
                if var2 not in self.transitions[var1]:
                    self.transitions[var1].append(var2)
        
        #write_to_csv(self.transitions, 'state_transition')
        with open('transitions.json', 'w') as f:
            json.dump(self.transitions, f)

    def test(self, file_paths):
        '''
        file_paths -> list: 
            complete path to the sample data files (.npy)
        '''
        if 'transitions.json' in os.listdir():
            with open('transitions.json', 'r') as f:
                transitions = json.load(f)
        else:
            raise(RuntimeError('Transition table missing'))
        anomalies = []
        for sample_path in file_paths:
            sample_data = load_sample(sample_path)

            for event1, event2 in zip(sample_data[0:-1], sample_data[1:]):
                #print(event1,event2)
                var1, var2 = event1[0], event2[0]
                ts1, ts2 = event1[1], event2[1]

                if var2 not in transitions[var1]:
                    print('Anomaly Detected:', [(var1, ts1), (var2, ts2), os.path.basename(sample_path)])
                    anomalies += [[(var1, ts1), (var2, ts2), os.path.basename(sample_path)]]
        return anomalies