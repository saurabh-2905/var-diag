from collections import defaultdict
import numpy as np
import os
from libraries.utils import read_traces, load_sample, write_to_csv
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
            if sample_path.find('.npy') != -1:
                sample_data = load_sample(sample_path)
                print(sample_path)
            elif sample_path.find('.json') != -1:
                sample_data = read_traces(sample_path)
                print(sample_path)
            else:
                sample_data = read_traces(sample_path)
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
            anomalies -> list: fromat: [[(var1, ts1), (var2, ts2), file_name], [], [], ...., []]
        '''
        if 'transitions_st.json' in os.listdir():
            with open('transitions_st.json', 'r') as f:
                transitions = json.load(f)
        else:
            raise(RuntimeError('Transition table missing'))
        
        ### convert the keys from string to int
        trans_keys = list(transitions.keys())
        transitions_new = {}
        for key in trans_keys:
            transitions_new[int(key)] = transitions[key]
        transitions = transitions_new
        print(transitions)

        anomalies = []
        for sample_path in file_paths:
            if sample_path.find('.npy') != -1:
                sample_data = load_sample(sample_path)
                print(sample_path)
            else:
                sample_data = read_traces(sample_path)
                print(sample_path)

            for event1, event2 in zip(sample_data[0:-1], sample_data[1:]):
                #print(event1,event2)
                var1, var2 = event1[0], event2[0]
                ts1, ts2 = event1[1], event2[1]
                # print(event1, event2)
                if var2 not in transitions[var1]:
                    print('Anomaly Detected:', [(var1, var2), (ts1, ts2), os.path.basename(sample_path)])
                    # anomalies += [[(var1, ts1), (var2, ts2), os.path.basename(sample_path)]]
                    anomalies += [[(var1, var2), (ts1, ts2), os.path.basename(sample_path)]]
        return anomalies
    
    def test_single(self, file_path):
        '''
        file_paths -> str: 
            complete path to the sample data file (.npy)
        '''

        if 'transitions_st.json' in os.listdir():
            with open('transitions_st.json', 'r') as f:
                transitions = json.load(f)
        else:
            raise(RuntimeError('Transition table missing'))
        
        ### convert the keys from string to int
        trans_keys = list(transitions.keys())
        transitions_new = {}
        for key in trans_keys:
            transitions_new[int(key)] = transitions[key]
        transitions = transitions_new
        print(transitions)

        anomalies = []
        if file_path.find('.npy') != -1:
            sample_data = load_sample(file_path)
            print(file_path)
        else:
            sample_data = read_traces(file_path)
            print(file_path)

        # sample_data  = sample_data[0:500]  ### get only first 500 events for testing
        for event1, event2 in zip(sample_data[0:-1], sample_data[1:]):
            #print(event1,event2)
            var1, var2 = event1[0], event2[0]
            ts1, ts2 = event1[1], event2[1]

            if var2 not in transitions[var1]:
                print('Anomaly Detected:', [(var1, var2), (ts1, ts2), os.path.basename(file_path)])
                # anomalies += [[(var1, ts1), (var2, ts2), os.path.basename(file_path)]]
                anomalies += [[(var1, var2), (ts1, ts2), os.path.basename(file_path)]]
        return anomalies
    

    def bbox_iou(self, b1_x1, b1_x2, b2_x1, b2_x2,):
        """
        Returns the IoU of two bounding boxes
        """

        # Get the coordinates of bounding boxes
        b1_y1 = b2_y1 = 0
        b1_y2 = b2_y2 = 1

        # get the corrdinates of the intersection rectangle
        inter_rect_x1 = max(b1_x1, b2_x1)
        inter_rect_y1 = max(b1_y1, b2_y1)
        inter_rect_x2 = min(b1_x2, b2_x2)
        inter_rect_y2 = min(b1_y2, b2_y2)
        # Intersection area
        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 , a_min=0, a_max=None) * np.clip(
            inter_rect_y2 - inter_rect_y1 , a_min=0, a_max=None)
        # Union Area
        b1_area = (b1_x2 - b1_x1 ) * (b1_y2 - b1_y1 )
        b2_area = (b2_x2 - b2_x1 ) * (b2_y2 - b2_y1 )
        print('inter:', inter_area)
        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou
    

    def get_correct_detections(self, detection, ground_truth):
        '''
        detection -> list: detections from the  -> [[(var1, 0), (ts1, ts2), file_name], [], [], ...., []]
        ground_truth -> list: ground truth labels -> [[(ind1, ind2), (ts1, ts2), class], [], [], ...., []]

        return:
        y_pred -> list: [1, 1, 0, 1, 0, 0, ...., 1]
        y_true -> list: [1, 1, 1, 1, 0, 0, ...., 0]
        '''
        
        gt_pred = defaultdict(list)      ### list of detections for each gt instance. The index of list denote its respective pred at that index, and list contains gt for that pred.
        rest_pred = [] ### list of detections that are not associated with any gt instance
        correct_pred = [] ### list of correct predictions
        y_true_ind = []
        y_pred_ind = []
        y_true = []
        y_pred = []
        # print('gt_pred:', gt_pred)
        # print(y_pred, y_true)

        if len(ground_truth) != 0:
            for gt_ind ,gt in enumerate(ground_truth):
                ind1 = gt[0]
                ind2 = gt[1]
                gt_ts1 = gt[2]
                gt_ts2 = gt[3]
                class_label = gt[4]

                if len(detection) != 0:
                    tmp_pred = []
                    for im, pred in enumerate(detection):
                        state1, state2 = pred[0]
                        pd_ts1, pd_ts2 = pred[1]
                        filename = pred[2]
                        # print('(gt_ts1, gt_ts2), (pd_ts1, pd_ts2)', (gt_ts1, gt_ts2), (pd_ts1, pd_ts2))

                        cond_1 = pd_ts1 >= gt_ts1 and pd_ts2 <= gt_ts2  ### check if the detection timestamp is within the ground truth timestamp (case 1)
                        cond_2 = pd_ts1 <= gt_ts1 and pd_ts2 >= gt_ts2  ### check if the gorund truth timestamp is within the detection timestamp (case 2)
                        cond_3 = pd_ts1 >= gt_ts1 and pd_ts1 <= gt_ts2 and pd_ts2 >= gt_ts2    ### partial detection on right of the ground truths, check 5 second difference after this (case 3)
                        cond_4 = pd_ts2 <= gt_ts2 and pd_ts2 >= gt_ts1 and pd_ts1 <= gt_ts1   ### partial detection on left of the ground truths, check 5 second difference after this (case 4)

                        
                        if cond_1 or cond_2 or cond_3 or cond_4:
                            print(gt_ind, im, cond_1, cond_2, cond_3, cond_4)
                            tmp_pred += [(im, pred, cond_1, cond_2, cond_3, cond_4)]    ### store all correct predictions that match with current gt      ### if cond_1 is TRUE, that means the detection is inside the gt and even multiple pred can be correct                  

                    if tmp_pred != []:
                        # print('tmp_pred', tmp_pred)
                        y_true_ind += [gt_ind]
                        ### if there are multiple correct predictions for a single gt, then choose the one that is closest to gt
                        if len(tmp_pred) > 1:
                            # print('if:', tmp_pred)
                            iou_pred = []
                            for ip, pred, case_1, case_2, case_3, case_4 in tmp_pred:
                                print('ip, pred', ip, pred)
                                state1, state2 = pred[0]
                                pd_ts1, pd_ts2 = pred[1]
                                filename = pred[2]
                                ### get IOU for all detections with gt
                                if not case_1:
                                    iou = self.bbox_iou(gt_ts1, gt_ts2, pd_ts1, pd_ts2)    ### calculate the IOU between the ground truth and the detection to evaluate which is the best detection for the given gt
                                else:
                                    iou = 1.0
                                iou_pred += [iou]

                            ### include all the detection with case1, even if there are multiple detections, as they are all correct
                            perfect_pred = False
                            for io, iou in enumerate(iou_pred):
                                if iou == 1.0:
                                    best_pred = tmp_pred[io]
                                    iou_pred[io] = 0
                                    if best_pred[0] not in y_pred_ind:
                                        perfect_pred = True
                                        y_pred_ind += [best_pred[0]]
                                        correct_pred += [best_pred[1]]
                                        y_true += [1]
                                        y_pred += [1]

                            if not perfect_pred:    ### skip selecting the best detection if there is a perfect detection (case 1)
                                best_pred_ind = iou_pred.index(max(iou_pred))
                                best_pred = tmp_pred[best_pred_ind]
                                print('ground_truth', gt)
                                print('best_pred:', best_pred)
                                print('y_pred_ind:', y_pred_ind)
                                # gt_pred[gt_ind] += [pred]  ### store the best detection for the given gt
                                if best_pred[0] not in y_pred_ind:
                                    y_pred_ind += [best_pred[0]]
                                    correct_pred += [best_pred[1]]
                                    y_true += [1]
                                    y_pred += [1]
                        else:
                            # print('else:', tmp_pred)
                            print('y_pred_ind:', y_pred_ind)
                            if tmp_pred[0][0] not in y_pred_ind:
                                y_pred_ind += [tmp_pred[0][0]]
                                correct_pred += [tmp_pred[0][1]]  
                                y_true += [1]
                                y_pred += [1]     
                    else:
                        ### this means no detection for this gt, denots FN
                        y_true += [1]
                        y_pred += [0]
            
        ### calculate FP
        for im, pred in enumerate(detection):
            if im not in y_pred_ind:
                rest_pred += [pred]
                y_true += [0]
                y_pred += [1]

        return correct_pred, rest_pred, y_pred, y_true
    
    

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