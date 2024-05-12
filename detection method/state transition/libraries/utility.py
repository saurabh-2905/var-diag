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


def plot_single_trace(df, var_list, with_time=False, anomalies=None, is_xticks=False):
    '''
    This function plots the traces with dataframe as input
    anomalies: list of anomalies -> list, format: ( list([start index, end index]), list([start timestamp, end timestamp]), list(class) )
    '''
    colour_list = ['lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral'] ### 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgrey', 'lightgreen', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow'
    class_list = ['normal', 'communication', 'sensor', 'bitflip']
    # Create figure
    fig = go.Figure()

    df_col = df.columns ### df_col = ['time', 'tracename']

    if with_time == False:
        fig.add_trace(
                    go.Scatter(y=list(df[df_col[1]]), name=df_col[1], mode='markers', marker=dict(size=10, color='midnightblue')),   ### equivalent to: y=list(df['trace1'])
                    )
    else:
        fig.add_trace(
                    go.Scatter(x=list(df[df_col[0]]), y=list(df[df_col[1]]), name=df_col[1], mode='markers', marker=dict(size=10, color='midnightblue')),   ### equivalent to: y=list(df['trace1'])
                    )
        
    ### plot anomalies
    if anomalies != None:
        ### sperate the content of anomalies
        anomalies_values = anomalies[0]
        anomalies_xticks = anomalies[1]
        anomalies_class = anomalies[2]

        for (start_ind, end_ind), (start_ts, end_ts), cls in zip(anomalies_values, anomalies_xticks, anomalies_class):
            ### check if time on x-axis
            if with_time:
                start = start_ts
                end = end_ts
            else:
                start = start_ind
                end = end_ind

            #### select colour based on class
            fill_colour = colour_list[cls]
            fig.add_shape(type="rect", # specify the shape type "rect"
                    xref="x", # reference the x-axis
                    yref="paper", # reference the y-axis
                    x0=start, # the x-coordinate of the left side of the rectangle
                    y0=0, # the y-coordinate of the bottom of the rectangle
                    x1=end, # the x-coordinate of the right side of the rectangle
                    y1=1, # the y-coordinate of the top of the rectangle
                    fillcolor=fill_colour, # the fill color
                    opacity=0.5, # the opacity
                    layer="below", # draw shape below traces
                    line_width=0, # outline width
                    )
            
            # Add dotted lines on the sides of the rectangle
            for x in [start, end]:
                fig.add_shape(type="line",
                        xref="x",
                        yref="paper",
                        x0=x,
                        y0=0,
                        x1=x,
                        y1=1,
                        line=dict(
                            color=fill_colour,
                            width=2,
                            dash="dot",
                        ),
                    )
        
        ##### add legend for anomaly based on colours/class
        for colour, name in zip(colour_list, class_list):
            fig.add_trace(go.Scatter(
                x=[None],  # these traces will not have any data points
                y=[None],
                mode='markers',
                marker=dict(size=10, color=colour),
                showlegend=True,
                name=name,
            ))
    ### generate x ticks with timestamp and index num  
    x_data = df[df_col[0]]
    if is_xticks == True and with_time == False:
        x_ticks = [(i,x_data[i]) for i in range(0,len(x_data),10) ]
        x_tickvals = [k for k in range(0,len(x_data),10)]
    elif is_xticks == True and with_time == True:
        x_ticks = [(i,x_data[i]) for i in range(0,len(x_data),10) ]
        x_tickvals = [x_data[k] for k in range(0,len(x_data),10)]
    elif is_xticks == False:
        x_ticks = None
        x_tickvals = None

    # Add range slider, title, yticks, axes labels
    fig.update_layout(
        title_text="Event Trace without Time",
        xaxis=dict(
            title="Number of events",
            rangeslider=dict(visible=True),
            type='linear',
            tickvals=x_tickvals,
            ticktext=x_ticks,
            tickfont = dict(size = FONTSIZE),
            titlefont = dict(size = FONTSIZE),
            color='black',
        ),
        yaxis=dict(
            title="Variables",
            tickvals=[k for k in range(1,len(var_list)+1)],
            ticktext= var_list,
            tickfont = dict(size = FONTSIZE),
            titlefont = dict(size = FONTSIZE),
            color='black',
        ),
        autosize=True,
        width=PLOTWIDTH,
        height=PLOTHEIGHT,
        plot_bgcolor='rgba(0,0,0,0)',
        
    )

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

    # style all the traces
    fig.update_traces(
        #hoverinfo="name+x+text",
        line={"width": 0.5},
        marker={"size": 8},
        mode="lines+markers",
        showlegend=True,
        
    )

    fig.show()
    # break