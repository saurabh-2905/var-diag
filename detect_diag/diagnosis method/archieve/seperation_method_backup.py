
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