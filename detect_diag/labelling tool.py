# This Python file uses the following encoding: utf-8
# __author__ = "Saurabh Band"

'''
- This is a GUI tool for generating labels for event traces generated using VarLogger. 
- This tools take the path to the trace file which is further used to map index to timestamps and also extract meta data for the trace
- The user needs to input the index1, index2 and class for the trace and the tool will calculate the timestamps for the given indices.
- The labels are stored in a dictionary with the key as the file name and the value as a list of labels for the trace.
- The labels provide indices and respective timestamp where anomaly has occured in the trace. 
- Even though we treat the anomalies as point anomalies, we mark a small window in the trace where the anomaly has occured using the indices.
- The detection algorithms can predict either the indices or the timestamps where the anomaly has occured.
The labels generated using the tools are tored in JSON format with the following structure:
{
    "metadata": {
        "behaviour": "behaviour of the trace",
        "version": "version of the trace",
        "thread": "thread of the trace",
        "code": "code of the trace",
        "path": "path to the trace file"
    },
    "labels": {
        "file_name": [
            [index1, index2, timestamp1, timestamp2, class],
            [index1, index2, timestamp1, timestamp2, class],
            [index1, index2, timestamp1, timestamp2, class]
        ]
    }
class: 0 -> normal, 1 -> comm anomaly, 2 -> sensor anomaly, 3 -> bitflip anomaly

'''

import json
import tkinter as tk
from tkinter import messagebox
import os

# Create root window
root = tk.Tk()

# To map the index to the timestamp
timestamps = []
dup_file = 0

# Get trace file to calculate timestamps from index
def generate_mapper():
    trace_path = tracefile_path.get()
    trace = read_traces(trace_path)
    for _, time in trace:
        timestamps.append(time)

    print('timestamps', timestamps)
    for i, (field, entry) in enumerate(metadata_entries.items()):
        if field == 'path':
            entry.insert(0, tracefile_path.get())
        else:
            i_new = i + 2
            entry.insert(0, tracefile_path.get().split('/')[-i_new])

    for field, entry in label_entries.items():
        if field == 'file_name':
            entry.insert(0, tracefile_path.get().split('/')[-1])
            

# Get path to trace file
frame = tk.Frame(root)
frame.pack()
label = tk.Label(frame, text='path to trace file')
label.pack(side='left')
entry = tk.Entry(frame)
entry.pack(side='right')
tracefile_path = entry

# Create save label button
generate_mapper_button = tk.Button(root, text='Generate Mapper', command=generate_mapper)
generate_mapper_button.pack()


metadata_entries = {}
for i, field in enumerate(['behaviour', 'version','thread', 'code', 'path']):
    i_new = i + 2
    print(tracefile_path.get())
    frame = tk.Frame(root)
    frame.pack()
    label = tk.Label(frame, text=field)
    label.pack(side='left')
    # entry = tk.Entry(frame)
    # metadata_entries[field] = entry
    entry = tk.Entry(frame)
    entry.pack(side='right')
    metadata_entries[field] = entry

# Create label entries
label_entries = {}
for field in ['file_name', 'index1', 'index2', 'class']:
    frame = tk.Frame(root)
    frame.pack()
    label = tk.Label(frame, text=field)
    label.pack(side='left')
    entry = tk.Entry(frame)
    entry.pack(side='right')
    label_entries[field] = entry

# Create labels list
labels = {}
trace = None

def read_traces(trace_path):
    '''
    read the trace files and extract variable names
    trace_path: path to the trace files -> str

    return:
    data = [ [event, timestamp], [], [],......,[] ]
    '''
    with open(trace_path, 'r') as f:
        data = json.load(f)
    return data



def calculate_timestamps(index1, index2):
    index1 = int(index1)
    index2 = int(index2)
    timestamp1 = timestamps[index1]
    timestamp2 = timestamps[index2]
    return timestamp1, timestamp2

# Function to save label
def save_label():
    file_name = label_entries['file_name'].get()
    label = tuple(entry.get() for field, entry in label_entries.items() if field != 'file_name')
    print(label)
    timestamp1, timestamp2 = calculate_timestamps(label[0], label[1])
    label = (int(label[0]), int(label[1]), timestamp1, timestamp2, int(label[2]))    ### get the labels in format (index1, index2 timestamp1, timestamp2, class)
    if file_name in labels:
        labels[file_name].append(label)
    else:
        labels[file_name] = [label]
    for field, entry in label_entries.items():
        if field != 'file_name' and field != 'class':   ### do not erase content of file_name so that we can store multiple labels for same file name
            entry.delete(0, 'end')

# Create save label button
save_label_button = tk.Button(root, text='Save label', command=save_label)
save_label_button.pack()

# Function to save data
def save_data():
    global dup_file

    answer = messagebox.askyesno(title='confirmation',
                    message='Are you sure that you want to save?')
    
    if answer:
        # Get metadata
        metadata = {field: entry.get() for field, entry in metadata_entries.items()}

        # Create data
        data = {
            'metadata': metadata,
            'labels': labels
        }

        # Write data to JSON file
        write_path = metadata_entries['path'].get().split('/')[-1]
        ### check if files already exists
        if os.path.exists(f'{write_path}_labels.json'):
            dup_file += 1
            with open(f'{write_path}_labels_{dup_file}.json', 'w') as f:
                json.dump(data, f, indent=4)
            # Show success message
            messagebox.showinfo('Success', f'{write_path}_labels_{dup_file}.json')
        else:
            with open(f'{write_path}_labels.json', 'w') as f:
                json.dump(data, f, indent=4)
            # Show success message
            messagebox.showinfo('Success', f'{write_path}_labels.json')

        # Clear all entries
        for field, entry in label_entries.items():
            entry.delete(0, 'end')

        for field, entry in metadata_entries.items():
            entry.delete(0, 'end')

        metadata = {}
        labels.clear()
        timestamps.clear()
    else:
        pass
# Create save button
save_button = tk.Button(root, text='Save data', command=save_data)
save_button.pack()

# Start main loop
root.mainloop()