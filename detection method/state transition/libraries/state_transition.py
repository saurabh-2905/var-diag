from collections import defaultdict
import numpy as np
import os
from libraries.utility import load_sample, write_to_csv, read_traces
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
        with open('transitions_st.json', 'w') as f:
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
    
    def test_single(self, file_path):
        '''
        file_paths -> str: 
            complete path to the sample data file (.npy)
        '''

        if 'transitions.json' in os.listdir():
            with open('transitions.json', 'r') as f:
                transitions = json.load(f)
        else:
            raise(RuntimeError('Transition table missing'))
        
        anomalies = []
        sample_data = load_sample(file_path)

        for event1, event2 in zip(sample_data[0:-1], sample_data[1:]):
            #print(event1,event2)
            var1, var2 = event1[0], event2[0]
            ts1, ts2 = event1[1], event2[1]

            if var2 not in transitions[var1]:
                print('Anomaly Detected:', [(var1, ts1), (var2, ts2), os.path.basename(file_path)])
                anomalies += [[(var1, ts1), (var2, ts2), os.path.basename(file_path)]]
        return anomalies
    

class StateTransitionProb:

    def __init__(self):
        self.transitions = defaultdict(list)  ### store all the vlaid transitions | key: "var1,var2", value: probability of tranisiton
        self.probabilities = defaultdict(list) ### store the probabilities of each transition | key: "var1,var2", value: [count, probability]
        self.invalid_transitions = defaultdict(list) ### store all the invalid transitions | key: "var1,var2", value: probability of tranisiton

        ### initialization of the nodes. Need this because these transitions also occur only once leading to low probability same as anomalous events
        self.initial_transitions = [
                    '1_0_main_ow,1_0_main_temp',
                    '1_0_main_temp,1_0_main_lora',
                    '1_0_main_lora,1_0_main_s',
                    '1_0_main_s,1_0_main_com_timer',
                    '1_0_main_com_timer,1_control_init_timer0_0',
                    '1_control_init_timer0_0,1_0_main_i'
                    ]

    def train(self, file_paths):
        '''
        file_paths -> list: 
            complete path to the trace files collected from nodes
        '''
        for sample_path in file_paths:
            sample_data = read_traces(sample_path)
            print(sample_path)
            total_events = len(sample_data)
            probabilities = defaultdict(list)
            for event1, event2 in zip(sample_data[0:-1], sample_data[1:]):
                # print(event1,event2)
                var1, var2 = event1[0], event2[0]
                #create a string "var1,var2" and give unique id for each unique var1, var2 pair
                var1_var2 = "{},{}".format(var1, var2)
                # print(var1, var2)
                if var1_var2 not in probabilities.keys():
                    #self.transitions[var1].append(var2)
                    probabilities[var1_var2] = [1, 0]  # Convert tuple to string
                else:
                    probabilities[var1_var2][0] += 1
            print(total_events)

            ### calculate probabilities for each transition and take average with last probability
            for key in probabilities.keys():
                probabilities[key][1] = probabilities[key][0] / total_events
                if key in self.probabilities.keys():
                    self.probabilities[key][1] = (self.probabilities[key][1] + probabilities[key][1]) / 2
                    self.probabilities[key][0] = (self.probabilities[key][0] + probabilities[key][0]) 
                else:
                    self.probabilities[key] = probabilities[key]

        #### calculate dynamic threshold
        threshold, keys = self.calculateDynamicThreshold()
                    
        ### check each tranisition against trheshold and detect anomalous transitions. Adapt the self.transition accordingly.
        ### convert tuple to string
        initial_transitions = [str(x) for x in self.initial_transitions]

        for key in keys:
            ### check if the transition is anomalous
            if self.probabilities[key][1] < threshold:
                ### check if the transition is in initial transitions
                if key not in initial_transitions:
                    self.invalid_transitions[key] = self.probabilities[key]
                else:
                    self.transitions[key] = self.probabilities[key]
            else:
                self.transitions[key] = self.probabilities[key]
                
        #write_to_csv(self.transitions, 'state_transition')
        with open('transitions_stp.json', 'w') as f:
            json.dump(self.transitions, f)

        with open('trans_probabilities.json', 'w') as f:
            json.dump(self.probabilities, f)


    def calculateDynamicThreshold(self, dynamic_threshold_factor=1.0):
        '''
        Calculate the dynamic threshold to seperate the normal and anomalous transitions

        dynamic_threshold_factor -> float: factor to scale the threshold based on the requirement. 
        '''
        keys = self.probabilities.keys()
        probs = []
        for key in keys:
            probs.append(self.probabilities[key][1])

        ### calculate dynamic threshold
        prob_mean = 1/len(keys)
        prob_variance = np.var(probs)
        prob_std = np.std(probs)
        dynamic_threshold = np.abs(prob_mean - dynamic_threshold_factor * prob_std)

        return dynamic_threshold, keys
    
    
    def test_single(self, file_path):
        '''
        file_paths -> str: 
            complete path to the sample data file
        '''

        if 'transitions_stp.json' in os.listdir():
            with open('transitions_stp.json', 'r') as f:
                transitions = json.load(f)
        else:
            raise(RuntimeError('Transition table missing'))
        
        anomalies = []
        sample_data = read_traces(file_path)

        for event1, event2 in zip(sample_data[0:-1], sample_data[1:]):
            #print(event1,event2)
            var1, var2 = event1[0], event2[0]
            ts1, ts2 = event1[1], event2[1]
            #create a string "var1,var2" and give unique id for each unique var1, var2 pair
            var1_var2 = "{},{}".format(var1, var2)

            if var1_var2 not in transitions.keys():
                print('Anomaly Detected:', [(var1, ts1), (var2, ts2), os.path.basename(file_path)])
                anomalies += [[(var1, ts1), (var2, ts2), os.path.basename(file_path)]]
            

        return anomalies
    

    def test(self, file_paths):
        '''
        file_paths -> list: 
            complete path to the sample data files
        '''

        if 'transitions_stp.json' in os.listdir():
            with open('transitions_stp.json', 'r') as f:
                transitions = json.load(f)
        else:
            raise(RuntimeError('Transition table missing'))
        
        anomalies_all = []
        for sample_path in file_paths:
            anomalies = self.test_single(sample_path)
            anomalies_all += anomalies

        return anomalies_all