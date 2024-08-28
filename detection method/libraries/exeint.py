from libraries.utils import load_sample, write_to_csv, read_traces
import json
import os
import numpy as np
from scipy.stats import t
from collections import defaultdict

import plotly.graph_objects as go

from sklearn.neighbors import LocalOutlierFactor




class exeInt:
    def __init__(self):
        pass


    def calculate_confidence_interval(self, data, confidence=0.95):
        '''
        calculate the confidence interval of the data
        data: a list of execution intervals -> [1,2,3,4,5,6,7,8,9,10]
        '''
        n = len(data)
        m = np.mean(data)
        std_err = np.std(data, ddof=1) / np.sqrt(n)
        h = std_err * t.ppf((1 + confidence) / 2, n - 1)
        start = m - h
        end = m + h
        return start, end
    
    
    def get_confinterval(self, exe_list):
        '''
        exe_list: dictionary containing the execution intervals for each variable for all files -> dict

        return:
        confidence_intervals: dictionary containing the confidence intervals for each variable -> dict
        '''
        confidence_intervals = {}
        for key in exe_list.keys():
            data = exe_list[key]
            start, end = self.calculate_confidence_interval(data)
            confidence_intervals[key] = [start, end]
        return confidence_intervals
    

    def get_dynamicthresh(self, exe_list):
        '''
        exe_list: dictionary containing the execution intervals for each variable for all files -> dict

        return:
        thresholds: dictionary containing the threshold values for each variable -> dict
        '''
        ### get uniques values from exe_list
        unique_values = {}
        outliers = {}
        for key in exe_list.keys():
            data = exe_list[key]
            data = [round(x, 1) for x in data]  ### round the values to 1 decimal point
            unique_values[key] = list(set(data))
            ### calculate probability for each unique value
            prob = {}
            for val in unique_values[key]:
                prob[val] = data.count(val)/len(data)
            unique_values[key] = prob

        ### consider values with probability > 0.05
        outliers[key] = dict()
        for key in unique_values.keys():
            print(key)
            prob = unique_values[key]
            # print(prob.keys())
            filtered_values = defaultdict(list)
            out = dict()
            for val in prob.keys():
                print('value:', val, 'prob:', prob[val])
                if prob[val] > 0.00:    ### 0.009 based on mamba dataset, to avoid the starting exeinterval which is outlier
                    filtered_values[val] = prob[val]
                else:
                    out[val] = prob[val]


            unique_values[key] = filtered_values
            outliers[key] = out


        ### get upper and lower bound by taking min and max from unique_values (can try some other approach)
        thresholds = {}
        for key in unique_values.keys():
            # print('Unique values:', key, unique_values[key])
            values = list(unique_values[key].keys())
            # print('Key:', key, len(values))
            if len(values) >= 1:
                thresholds[key] = [np.clip(np.round(min(values)-0.1, 1), 0, None), np.clip(np.round(max(values)+0.1, 1), 0, None)]

        return thresholds

    def train_lof(self, exe_list):
        '''
        exe_list: dictionary containing the execution intervals for each variable for all files -> dict

        return:
        lof_models: dictionary containing the lof models for each variable -> dict
        '''
        lof_models = {}
        for key in exe_list.keys():
            # print(key)
            x_data = exe_list[key]
            x_data = [round(x, 1) for x in x_data]
            x_data = np.array(x_data).reshape(-1, 1)
            # print(x_data)

            lof = LocalOutlierFactor(n_neighbors=2, contamination=0.01, novelty=True).fit(x_data)
            lof_models[key] = lof

        return lof_models

    def get_exeint(self, train_data_path):
        '''
        train_data_path: list of paths to the training trace files -> list

        return:
        exe_list: dictionary containing the execution intervals for each variable -> dict
        filewise_exe_list: dictionary containing the execution intervals for each variable for each file -> dict
        '''
        exe_list = {}   ### {var1: [1,2,3,4,5,6,7,8,9,10], var2: [1,2,3,4,5,6,7,8,9,10], ....}
        filewise_exe_list = {}   ### {file1: {var1: [1,2,3,4,5,6,7,8,9,10], var2: [1,2,3,4,5,6,7,8,9,10], ....}, file2: {var1: [1,2,3,4,5,6,7,8,9,10], var2: [1,2,3,4,5,6,7,8,9,10], ....}, ....}
        for sample_path in train_data_path:
            print(sample_path)
            # sample_data = read_traces(sample_path)
            if sample_path.find('.npy') != -1:
                sample_data = load_sample(sample_path)
                print(sample_path)
            elif sample_path.find('.json') != -1:
                sample_data = read_traces(sample_path)
                print(sample_path)
            else:
                sample_data = read_traces(sample_path)
                print(sample_path)
            filename = sample_path.split('/')[-1].split('.')[0]
            # print(sample_data)
            ### collect timestamps for all variables
            timestamps = {}
            for i, event in enumerate(sample_data):
                var, ts = event
                ts = int(ts)
                # print(var, ts)
                if var not in timestamps.keys():
                    timestamps[var] = [ts]
                else:
                    timestamps[var].append(ts)

            print(timestamps.keys())
            ### calculate execution intervals for all variables
            intervals = {}
            for key in timestamps.keys():
                ts_list = timestamps[key]
                for ts1, ts2 in zip(ts_list[:-1], ts_list[1:]):
                    exe_time = ts2 - ts1
                    ### convert timestampt from miliseconds to seconds, and only consdider 1 decimal point. 
                    exe_time = round(exe_time/1000, 2)
                    print(key, ts1,ts2)

                    ### for filewise exe_list
                    if key not in intervals.keys():
                        intervals[key] = [exe_time]
                    else:
                        intervals[key].append(exe_time)

                    ### overall exe list
                    if key not in exe_list.keys():
                        exe_list[key] = [exe_time]
                    else:
                        exe_list[key].append(exe_time)

                # ### remove the variable if it has less than 3 execution intervals to avoid problem with lof
                # for k in intervals.keys():
                #     cont_chk = intervals[k]
                #     if len(cont_chk) <= 3:  
                #         del intervals[k]

            filewise_exe_list[filename] = intervals
        
        ### remove the variable if it has less than 3 execution intervals to avoid problem with lof
        for k in timestamps.keys():
            if k in exe_list.keys():
                cont_chk = exe_list[k]
                if len(cont_chk) <= 3:  
                    del exe_list[k]
                    exe_list[k] = [0.0, 0.0, 0.0]    ### add 0.0 to avoid problem with lof
            else:
                exe_list[k] = [0.0, 0.0, 0.0]

        return exe_list, filewise_exe_list


    def test_single(self, sample_path, thresholds=None, lof_models=None):
        '''
        sample_path: path to the test trace file -> str
        thresholds: dictionary containing the threshold values for each variable -> dict

        return:
        detected_anomalies: list of detected anomalies -> list
        '''
        detected_anomalies = []
        sample_data = read_traces(sample_path)
        # sample_data  = sample_data[0:500]    ### get only first 500 events for testing
        filename = sample_path.split('/')[-1].split('.')[0]
        ### iterate trace and make decision for each exe interval
        var_tracking = {}
        for i in range(len(sample_data)):
            event = sample_data[i]
            var, ts = event
            ts = int(ts)
            if var not in var_tracking.keys():
                var_tracking[var] = [ts]
            else:
                var_tracking[var].append(ts)

            ### calculate exe interval
            if len(var_tracking[var]) > 3:
                exe_time = var_tracking[var][-1] - var_tracking[var][-2]
                ### convert timestampt from miliseconds to seconds, and only consdider 1 decimal point. 
                exe_time = round(exe_time/1000, 1)

                if thresholds != None:
                    ### check if var was present in the training data
                    # if var not in thresholds.keys():
                    #     thresholds[var] = [0.0, 3]
                        
                    if var in thresholds.keys():
                        ### check if exe_time is an outlier
                        if exe_time < thresholds[var][0] or exe_time > thresholds[var][1]:
                            print(f'Anomaly detected for {var} in {filename} at {i}th event')
                            print(f'Execution interval: {exe_time}')
                            detected_anomalies += [[(var,0), (var_tracking[var][-2], var_tracking[var][-1]), os.path.basename(sample_path)]]    ### 0 in (var,0) is to keep the detection format same as ST 
                            # detected_anomalies += [[(var,0), (var_tracking[var][-1]-5000, var_tracking[var][-1]), os.path.basename(sample_path)]]    ### 0 in (var,0) is to keep the detection format same as ST 
                        else:
                            print(f'{var} not present during training')
                
                if lof_models != None:
                    ### check if exe_time is an outlier
                    if var not in lof_models.keys():
                        print(f'Anomaly detected for {var} in {filename} at {i}th event')
                        print(f'Execution interval: {exe_time}')
                        # detected_anomalies += [[(var, var_tracking[var][-2]), (var, var_tracking[var][-1]), os.path.basename(sample_path)]]
                        detected_anomalies += [[(var,0), (var_tracking[var][-1]-3000, var_tracking[var][-1]), os.path.basename(sample_path)]]
                    else:
                        lof = lof_models[var]
                        test_data = np.array([exe_time])
                        # print(test_data)
                        detection = lof.predict([test_data])
                        # print('detection:', detection)
                        if detection == -1:
                            print(f'Anomaly detected for {var} in {filename} at {i}th event')
                            print(f'Execution interval: {exe_time}')
                            # detected_anomalies += [[(var, var_tracking[var][-2]), (var, var_tracking[var][-1]), os.path.basename(sample_path)]]
                            detected_anomalies += [[(var,0), (var_tracking[var][-2], var_tracking[var][-1]), os.path.basename(sample_path)]]

        return detected_anomalies
    

    def remove_duplicates(self, detections, diff_val=2):
        '''
        This fucntions removes multiple detections that are caused because of multiple variables
        Each anomaly that occurs, affects group of variables, resulting in multiple detections for single ground truth
        This function groups the detections based on the time difference between them and selects one from each group

        detections: list of detected anomalies -> list -> in format: [[(var1, 0), (ts1, ts2), filename], [(var2, 0), (ts1, ts2), filename], ....]
        diff_val: time difference threshold in seconds to group detections -> int/float

        return:
        dedup_detection: list of deduplicated detections -> list
        '''
        DIFF_VAL = diff_val
        pred = detections
        ### sort the list using the first timestamp of every detection
        pred = sorted(pred, key=lambda x: x[1][0])
        print('sorted detecions:', pred)
        det_ts1 = [ x[1][0]/1000 for x in pred]  ### get first timestamp of every detection and convert from mili second to second
        print(det_ts1)
        group = []
        group_ind = []
        aggregated_ts = []
        aggregated_ts_ind = []
        for xi, (x1, x2) in enumerate(zip(det_ts1[0:-1], det_ts1[1:])):
            # print(xi)
            diff_ts = abs(x2 - x1)
            # print(diff_ts, x1, x2)
            ### decision to wether or not group detections
            if diff_ts < DIFF_VAL:
                group += [pred[xi]]
                group_ind += [xi]
                if xi == len(det_ts1)-2:  ### for last pair
                    group += [pred[xi+1]]
                    group_ind += [xi+1]
                    aggregated_ts_ind += [group_ind]
                    aggregated_ts += [group]
            elif diff_ts >= DIFF_VAL:
                group_ind += [xi]
                group += [pred[xi]]   ### group the predictions which have time diff less than 2 seconds
                # print(group)
                aggregated_ts_ind += [group_ind]   
                aggregated_ts += [group]    ### collect all the groups
                group = []
                if xi == len(det_ts1)-2:   ### for last pair
                    group += [pred[xi+1]]
                    group_ind += [xi+1]
                    aggregated_ts_ind += [group_ind]
                    aggregated_ts += [group]

        ### de-duplicate the detections (select one from each group)
        dedup_detection = []
        for gp in aggregated_ts:
            # dedup_detection += [gp[0]]    ### take first detection from each group
            select_sample = len(gp)//2
            dedup_detection += [gp[select_sample]]

        return dedup_detection, aggregated_ts
    

    def merge_detections(self, detections, diff_val=2):
        '''
        This fucntions merges multiple detections that are less the 2 seconds apart. 
        These multiple detections can be caused because of multiple variables or even multiple anomalies that are colser
        Each anomaly that occurs, affects group of variables, resulting in multiple detections for single ground truth
        This function groups the detections based on the time difference between them and selects one from each group

        detections: list of detected anomalies -> list -> in format: [[(var1, 0), (ts1, ts2), filename], [(var2, 0), (ts1, ts2), filename], ....]
        diff_val: time difference threshold in seconds to group detections -> int/float

        return:
        dedup_detection: list of deduplicated detections -> list
        '''
        DIFF_VAL = diff_val
        pred = detections
        ### sort the list using the first timestamp of every detection
        pred = sorted(pred, key=lambda x: x[1][0])
        print('sorted detecions:', pred)
        det_ts1 = [ x[1][0]/1000 for x in pred]  ### get first timestamp of every detection and convert from mili second to second
        det_ts2 = [ x[1][1]/1000 for x in pred]  ### get first timestamp of every detection and convert from mili second to second
        # print('merge ts:', pred[0][1], det_ts1[0], det_ts2[0])
        group = []
        group_ind = []
        aggregated_ts = []
        aggregated_ts_ind = []
        for xi, (x1, x2, y1, y2) in enumerate(zip(det_ts1[0:-1], det_ts1[1:], det_ts2[0:-1], det_ts2[1:])):    ### get the first and last timestamp of every detection
            # print(xi)
            diff_ts1 = abs(x2 - x1)
            # diff_ts2 = abs(y2 - y1)    ### did not use it because the problem was solved by increasing diff_vaal to 5
            print('Merge diff:', diff_ts1, x1, x2)
            ### decision to wether or not group detections
            if diff_ts1 <= DIFF_VAL:
                group += [pred[xi]]
                group_ind += [xi]
                if xi == len(det_ts1)-2:  ### for last pair
                    group += [pred[xi+1]]
                    group_ind += [xi+1]
                    aggregated_ts_ind += [group_ind]
                    aggregated_ts += [group]
            elif diff_ts1 > DIFF_VAL:
                group_ind += [xi]
                group += [pred[xi]]   ### group the predictions which have time diff less than 2 seconds
                # print(group)
                aggregated_ts_ind += [group_ind]   
                aggregated_ts += [group]    ### collect all the groups
                group = []
                if xi == len(det_ts1)-2:   ### for last pair
                    group += [pred[xi+1]]
                    group_ind += [xi+1]
                    aggregated_ts_ind += [group_ind]
                    aggregated_ts += [group]

        ### merge the detections (starting TS from first detection and ending TS from last detection from each group)
        merge_detection = []
        for gp in aggregated_ts:    #### read single group of detections. in format list of detections
            ### sort the detections in ascending order based on the first timestamp
            gp = sorted(gp, key=lambda x: x[1][0])
            #### scan through the group and get the lowest first timestamp and highest last timestamp
            lowest_ts = gp[0][1][0]
            highest_ts = gp[-1][1][1]
            first_event = gp[0][0][0]
            last_event = gp[-1][0][0]
            for i, (_, gts, _) in enumerate(gp):
                if gts[0] < lowest_ts:
                    lowest_ts = gts[0]
                    first_event = gp[i][0][0]
                if gts[1] > highest_ts:
                    highest_ts = gts[1]
                    last_event = gp[i][0][0]


            merge_detection += [[(first_event, last_event), (lowest_ts, highest_ts), gp[0][2]]]    
            ### eg. [(10, 8), (27249, 30675), 'trace2-sensor'], where first tuple contains variable of first and last detection in group, second tuple is the first TS of first detection and second ts of last detection, and the file name is taken from first detection

        return merge_detection, aggregated_ts


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
        false_negatives = [] ### list of false negatives
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
                        false_negatives += [gt]
            
        ### calculate FP
        for im, pred in enumerate(detection):
            if im not in y_pred_ind:
                rest_pred += [pred]
                y_true += [0]
                y_pred += [1]

        return correct_pred, rest_pred, y_pred, y_true, false_negatives
    

    def viz_thresholds(self, exe_list, confidence_intervals=None, thresholds=None):
        for key in exe_list.keys():
            fig = go.Figure()

            # Histogram
            fig.add_trace(go.Histogram(x=exe_list[key], nbinsx=100, name='execution intervals', histnorm='probability', marker=dict(color='midnightblue')))

            if confidence_intervals != None:
                # Vertical lines
                fig.add_shape(type="line", x0=confidence_intervals[key][0], x1=confidence_intervals[key][0], y0=0, y1=1, yref='paper', line=dict(color="Red", dash="dash"))
                fig.add_shape(type="line", x0=confidence_intervals[key][1], x1=confidence_intervals[key][1], y0=0, y1=1, yref='paper', line=dict(color="Red", dash="dash"))
                # Add traces for the lines to include them in the legend
                fig.add_trace(go.Scatter(x=[confidence_intervals[key][0]], y=[0], mode='lines', name='Confidence Interval', line=dict(color="Red", dash="dash"), showlegend=True))
            
            if thresholds != None:
                fig.add_shape(type="line", x0=min(thresholds[key]), x1=min(thresholds[key]), y0=0, y1=1, yref='paper', line=dict(color="Green", dash="dash"))
                fig.add_shape(type="line", x0=max(thresholds[key]), x1=max(thresholds[key]), y0=0, y1=1, yref='paper', line=dict(color="Green", dash="dash"))
                fig.add_trace(go.Scatter(x=[min(thresholds[key])], y=[0], mode='lines', name='Dynamic Threshold', line=dict(color="Green", dash="dash"), showlegend=True))

            # Layout
            fig.update_layout(title=key, xaxis_title="Value", yaxis_title="Count", bargap=0.2, bargroupgap=0.1, title_font_size=20,
                                xaxis=dict(
                                    tickfont = dict(size = 20),
                                    titlefont = dict(size = 20),
                                    color='black',
                                ),
                                yaxis=dict(
                                    tickfont = dict(size = 20),
                                    titlefont = dict(size = 20),
                                    color='black'
                                ),
                                plot_bgcolor='rgba(0,0,0,0)',)
            
            fig.update_xaxes(
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='black',
                gridcolor='lightgrey'
            )
            
            fig.update_yaxes(
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='black',
                gridcolor='lightgrey'
            )

            fig.show()
