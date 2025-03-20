
################################
### Function for method 1 and 2
################################

def strip_correct_part(ref_samples, test_events, test_intervals, test_timestamps):
    '''
    check if any matching event trace in present based on first 2 points
    if yes, then check the number of matching events and intervals, remove the matching part in the event trace and return the remaining part
    if no, then return the same event trace
    '''

    test_data_len = len(test_events)
    # print('test_events:', test_events)
    print('test_data_len:', test_data_len)
    ### shortlist the reference samples which has first 5 elements same as the test_trace
    shortlisted_ref_events = []
    shortlisted_ref_intervals = []
    zero_count = []
    for ref_sample in ref_samples:
        # print('ref_sample:', ref_sample[0][:5])
        # event_diff = np.array(ref_sample[0][0:len(test_events)]) - np.array(test_events)
        # print('event_diff:', event_diff)
        # print('event_diff:', len(event_diff))
        # print('zeros:', np.where(event_diff==0)[0].shape)
        # if len(test_events) == 276:
        #     print('ref_sample:', ref_sample[0][:5])
        #     print('test_events:', test_events[:5])
        if ref_sample[0][:2] == test_events[:2]:
            ref_events = ref_sample[0][:test_data_len]
            ref_intervals = ref_sample[1][:test_data_len]
            diff_events = np.array(ref_events) - np.array(test_events)
            # print('len:', len(ref_intervals), len(test_intervals))
            diff_intervals = np.abs(np.array(ref_intervals) - np.array(test_intervals))
            if len(test_events) == 276:
                print('diff_events:', diff_events)
                print('diff_intervals:', diff_intervals)
            count = 0
            # print(sf[0], sf[1])
            for esf, esi in zip(diff_events, diff_intervals):
                ### check if events and intervals are same
                # if esf == 0 and esi < 5:
                if esf == 0:
                    count += 1
                else:
                    break   ### part of the logic, do not remove

            # print('zero_count:', count)
            ### depulicate the ref samples
            if ref_events not in shortlisted_ref_events:
                zero_count.append(count)
                shortlisted_ref_events.append(ref_events)
                shortlisted_ref_intervals.append(ref_intervals)
            # print('count:', count)  

        # break

    print('zero_count:', zero_count)
    print('shortlisted_ref_samples:', len(shortlisted_ref_events))

    ### select the ref_sample_events with maximum leading zeros
    if len(zero_count) != 0:
        max_zero_count = max(zero_count)
        zero_count = np.array(zero_count)
        max_zero_count_ind = np.where(zero_count==max_zero_count)[0][0]
        # print('max_zero_count_ind:', max_zero_count_ind)
        selected_ref_events = shortlisted_ref_events[max_zero_count_ind]
        selected_ref_intervals = shortlisted_ref_intervals[max_zero_count_ind]
        # print('selected_ref_events:', selected_ref_events[:max_zero_count+1])
    else:
        max_zero_count = 0
        selected_ref_events = None
        selected_ref_intervals = None

    if max_zero_count == 0:
        print('No match found')
        return test_events, test_intervals, test_timestamps, None, None, None
    else:
        ### select the point where the last match happened
        last_matched_point = max_zero_count-1
        # print('last_matched_point:', last_matched_point)
        # print('test_events:', test_events[:last_matched_point])
        striped_test_events = test_events[last_matched_point:]
        striped_test_intervals = test_intervals[last_matched_point:]
        striped_test_timestamps = test_timestamps[last_matched_point:]
        # print('striped_test_events:', striped_test_events)
        print('max count:', max_zero_count, len(test_events), len(striped_test_events))

        
        if max_zero_count == len(test_events):
            print('All events are same')
            return [], None, None, None, None, None
        else:
            return striped_test_events, striped_test_intervals, striped_test_timestamps, max_zero_count, selected_ref_events, selected_ref_intervals

############################################################################################################





#### method 1

### store the first to consecutive points as the first anomaly instance
an_instance = []
an_timestamps = []
all_instances = []
all_timestamps = []
all_ref_events = []
all_ref_intervals = []
all_striped_test_events = []
all_striped_test_intervals = []
all_striped_timestamps = []

striped_test_events = test_events
striped_test_intervals = test_intervals
striped_timestamps = test_timestamps
print('striped_test_events:', striped_test_events, len(striped_test_events))
# print('striped_test_intervals:', striped_test_intervals, len(striped_test_intervals))
i = 0
first_loop = True
while len(striped_test_events) > 0:
# while i < 6:
    ### first collect the trace that is given as input
    all_striped_test_events.append(striped_test_events)
    all_striped_test_intervals.append(striped_test_intervals)
    all_striped_timestamps.append(striped_timestamps)
    ### remove the initial correct part of the trace
    striped_test_events, striped_test_intervals, striped_timestamps, max_zero_count, selected_ref_events, selected_ref_intervals = strip_correct_part(ref_samples, striped_test_events, striped_test_intervals, striped_timestamps)
    print('striped_test_events 1:', striped_test_events, len(striped_test_events))
    ### get the ref trace that matched with the test trace
    all_ref_events.append(selected_ref_events)
    all_ref_intervals.append(selected_ref_intervals)

    # break
    ### store the first anomaly instance (first two consecutive points)
    if max_zero_count != None:
        if len(an_instance) != 0:
            ### new anomaly instance detected
            all_instances.append(an_instance)
            all_timestamps.append(an_timestamps)
            an_instance = []
            an_timestamps = []
        ### start, where first anomaly instance is detected
        an_instance.extend(striped_test_events[:2])
        an_timestamps.extend(striped_timestamps[:2])
        striped_test_events = striped_test_events[1:]
        striped_test_intervals = striped_test_intervals[1:]
        striped_timestamps = striped_timestamps[1:]
        print('striped_test_events 2:', striped_test_events, len(striped_test_events))
    else:
        if len(striped_test_events) > 0:
            ### normal functionality loop
            an_instance.extend(striped_test_events[1:2])
            an_timestamps.extend(striped_timestamps[1:2])
            striped_test_events = striped_test_events[1:]
            striped_test_intervals = striped_test_intervals[1:]
            striped_timestamps = striped_timestamps[1:]
            print('striped_test_events 3:', striped_test_events, len(striped_test_events))
        else:
            ### for last iteration, when len(striped_test_events) == 1   
            all_instances.append(an_instance)
            all_timestamps.append(an_timestamps)
            anomaly_instances.append(all_instances)
            anomaly_timestamps.append(all_timestamps)
            if len(striped_test_events) == 0:
                break
    # print('an_instance:', an_instance)
    # print('')    
    i += 1



#### method 2
an_instance = []
an_timestamps = []
all_instances = []
all_timestamps = []
all_ref_events = []
all_ref_intervals = []
all_striped_test_events = []
all_striped_test_intervals = []
all_striped_timestamps = []

striped_test_events = test_events
striped_test_intervals = test_intervals
striped_timestamps = test_timestamps
print('striped_test_events:', striped_test_events, len(striped_test_events))
# print('striped_test_intervals:', striped_test_intervals, len(striped_test_intervals))
i = 0
first_loop = True
while len(striped_test_events) > 0:
# while i < 6:
    ### first collect the trace that is given as input
    all_striped_test_events.append(striped_test_events)
    all_striped_test_intervals.append(striped_test_intervals)
    all_striped_timestamps.append(striped_timestamps)
    ### remove the initial correct part of the trace
    striped_test_events, striped_test_intervals, striped_timestamps, max_zero_count, selected_ref_events, selected_ref_intervals = strip_correct_part(ref_samples, striped_test_events, striped_test_intervals, striped_timestamps)
    print('striped_test_events 1:', striped_test_events, len(striped_test_events))
    ### get the ref trace that matched with the test trace
    all_ref_events.append(selected_ref_events)
    all_ref_intervals.append(selected_ref_intervals)

    # break
    ### store the first anomaly instance (first two consecutive points)
    if max_zero_count != None:
        if len(an_instance) != 0:
            ### new anomaly instance detected
            all_instances.append(an_instance)
            all_timestamps.append(an_timestamps)
            an_instance = []
            an_timestamps = []
            first_loop = True
        # ### start, where first anomaly instance is detected
        # an_instance.extend(striped_test_events[:2])
        # an_timestamps.extend(striped_timestamps[:2])
        # striped_test_events = striped_test_events[1:]
        # striped_test_intervals = striped_test_intervals[1:]
        # striped_timestamps = striped_timestamps[1:]
        print('striped_test_events 2:', striped_test_events, len(striped_test_events))
    else:
        if len(striped_test_events) > 0:
            ### normal functionality loop
            if first_loop:
                an_instance.extend(striped_test_events[:2])
                an_timestamps.extend(striped_timestamps[:2])
                striped_test_events = striped_test_events[1:]
                striped_test_intervals = striped_test_intervals[1:]
                striped_timestamps = striped_timestamps[1:]
                first_loop = False
                print('striped_test_events 3.1:', striped_test_events, len(striped_test_events))
            else:
                an_instance.extend(striped_test_events[1:2])
                an_timestamps.extend(striped_timestamps[1:2])
                striped_test_events = striped_test_events[1:]
                striped_test_intervals = striped_test_intervals[1:]
                striped_timestamps = striped_timestamps[1:]
                print('striped_test_events 3.2:', striped_test_events, len(striped_test_events))
        else:
            ### for last iteration, when len(striped_test_events) == 1   
            if len(an_instance) != 0:
                all_instances.append(an_instance)
                all_timestamps.append(an_timestamps)
            anomaly_instances.append(all_instances)
            anomaly_timestamps.append(all_timestamps)
            if len(striped_test_events) == 0:
                break
    # print('an_instance:', an_instance)
    # print('')    
    i += 1



    #### method 3
    an_instance = []
    an_timestamps = []
    all_instances = []
    all_timestamps = []
    all_ref_events = []
    all_ref_intervals = []
    all_striped_test_events = []
    all_striped_test_intervals = []
    all_striped_timestamps = []

    striped_test_events = test_events
    striped_test_intervals = test_intervals
    striped_timestamps = test_timestamps
    # print('striped_test_events:', striped_test_events, len(striped_test_events))
    # print('striped_test_intervals:', striped_test_intervals, len(striped_test_intervals))

    feature_event, feature_inervals, feature_time_stamps = split_instances(ref_samples, striped_test_events, striped_test_intervals, striped_timestamps)
    print('feature_event:', feature_event)
    print('len:', len(feature_event))
        
         
    print('')
    # break