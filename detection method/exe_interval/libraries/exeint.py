from libraries.utils import load_sample, write_to_csv, read_traces
import json
import os
import numpy as np
from scipy.stats import t
from collections import defaultdict

import plotly.graph_objects as go




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
            print(prob.keys())
            filtered_values = defaultdict(list)
            out = dict()
            for val in prob.keys():
                print(prob[val])
                if prob[val] > 0.05:    
                    filtered_values[val] = prob[val]
                else:
                    out[val] = prob[val]


            unique_values[key] = filtered_values
            outliers[key] = out


        ### get upper and lower bound by taking min and max from unique_values (can try some other approach)
        thresholds = {}
        for key in unique_values.keys():
            values = list(unique_values[key].keys())
            thresholds[key] = [round(min(values)-0.1, 1), round(max(values)+0.1, 1)]

        return thresholds



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
            sample_data = read_traces(sample_path)
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

            filewise_exe_list[filename] = intervals

            return exe_list, filewise_exe_list


    def test_single(self, sample_path, thresholds):
        '''
        sample_path: path to the test trace file -> str
        thresholds: dictionary containing the threshold values for each variable -> dict

        return:
        detected_anomalies: list of detected anomalies -> list
        '''
        detected_anomalies = []
        sample_data = read_traces(sample_path)
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
            if len(var_tracking[var]) > 1:
                exe_time = var_tracking[var][-1] - var_tracking[var][-2]
                ### convert timestampt from miliseconds to seconds, and only consdider 1 decimal point. 
                exe_time = round(exe_time/1000, 1)

                ### check if exe_time is an outlier
                if exe_time < thresholds[var][0] or exe_time > thresholds[var][1]:
                    print(f'Anomaly detected for {var} in {filename} at {i}th event')
                    print(f'Execution interval: {exe_time}')
                    # detected_anomalies += [[(var, var_tracking[var][-2]), (var, var_tracking[var][-1]), os.path.basename(sample_path)]]
                    detected_anomalies += [[(var,0), (var_tracking[var][-2], var_tracking[var][-1]), os.path.basename(sample_path)]]    ### 0 in (var,0) is to keep the detection format same as ST 

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
            dedup_detection += [gp[0]]    ### take first detection from each group

        return dedup_detection, aggregated_ts
    

    def viz_thresholds(self, exe_list, confidence_intervals, thresholds):
        for key in exe_list.keys():
            fig = go.Figure()

            # Histogram
            fig.add_trace(go.Histogram(x=exe_list[key], nbinsx=100, name='execution intervals', histnorm='probability', marker=dict(color='midnightblue')))

            # Vertical lines
            fig.add_shape(type="line", x0=confidence_intervals[key][0], x1=confidence_intervals[key][0], y0=0, y1=1, yref='paper', line=dict(color="Red", dash="dash"))
            fig.add_shape(type="line", x0=confidence_intervals[key][1], x1=confidence_intervals[key][1], y0=0, y1=1, yref='paper', line=dict(color="Red", dash="dash"))
            fig.add_shape(type="line", x0=min(thresholds[key]), x1=min(thresholds[key]), y0=0, y1=1, yref='paper', line=dict(color="Green", dash="dash"))
            fig.add_shape(type="line", x0=max(thresholds[key]), x1=max(thresholds[key]), y0=0, y1=1, yref='paper', line=dict(color="Green", dash="dash"))

            # Add traces for the lines to include them in the legend
            fig.add_trace(go.Scatter(x=[confidence_intervals[key][0]], y=[0], mode='lines', name='Confidence Interval', line=dict(color="Red", dash="dash"), showlegend=True))
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
