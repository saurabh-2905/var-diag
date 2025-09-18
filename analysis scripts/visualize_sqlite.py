import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, String, BigInteger, Sequence, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import List
from typing import Optional
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
import os
from libv3.utils import *
import pandas as pd
import json

import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

COMP_WIDTH = '50%'

############ Utility Functions ####################
def prepare_labels(paths_label):
    ### count and prepare labels to plot
    '''
    labels are of format [index1, index2, timestamp1, timestamp2, class]

    path_label: list of paths to label files    -> list

    return:
    toplot_gt: list of labels to plot           -> list
    '''
    class_count = defaultdict(int)
    for i, path in enumerate(paths_label):
        label_content = prepare_gt(path)
        ind, ts, cls = label_content
        # print(ind, ts, cls)
        for c in cls:
            class_count[c]+=1
            
        print(path)
        toplot_gt = label_content

        print(os.path.split(path)[-1], class_count)

        # break
    for key, val in class_count.items():
        print(key, val)
    
    return toplot_gt

def prepare_detections(paths_detection, timestamps):
    '''
    detections are of format [(state, state), (timestamp1, timestamp2), filename]

    path_detection: list of paths to detection files    -> list of str, len = 1
    timestamps: list of timestamps                      -> list of int

    return:
    plot_val: list of indices of detections                -> list of tuple
    plot_x_ticks: list of timestamps to plot x_ticks               -> list of tuple
    plot_class: list of class to plot                    -> list of int

    '''
    detections = read_traces(paths_detection[0])
    plot_val = []
    plot_x_ticks = []
    plot_class = []
    for det in detections:
        # print(det)
        det_ts1, det_ts2 = det[1]
        # print(det_ts1, det_ts2)

        det_ind1_pre = [ abs(t-det_ts1) for t in timestamps]
        det_ind1 = det_ind1_pre.index(min(det_ind1_pre))

        det_ind2_pre = [ abs(t-det_ts2) for t in timestamps]
        det_ind2 = det_ind2_pre.index(min(det_ind2_pre))
        # print(det_ind1, det_ind2)
        # print(timestamps[det_ind1], timestamps[det_ind2])

        plot_val += [(det_ind1, det_ind2)]
        plot_x_ticks += [(timestamps[det_ind1], timestamps[det_ind2])]
        plot_class += [0]

    return [plot_val, plot_x_ticks, plot_class]
############ Utility Functions ####################


############ AlchemySQL Database Structure ####################
# Define the base for the declarative model
Base = declarative_base()

# Define the Event class which will be mapped to the events table in the database
class Event(Base):
    __tablename__ = 'events'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    
    name: Mapped[int] = mapped_column(nullable=False)
    timestamp: Mapped[int] = mapped_column(BigInteger, nullable=False)
    file_number: Mapped[int] = mapped_column(ForeignKey("file_config.id")) 

    config: Mapped["File_config"] = relationship(back_populates="events")
    


class File_config(Base):
    __tablename__ = 'file_config'

    id: Mapped[int] = mapped_column(primary_key=True)
    unique_name: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    code_base: Mapped[str] = mapped_column(String(50), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    behaviour: Mapped[str] = mapped_column(String(50), nullable=False)
    trial_num: Mapped[int] = mapped_column(Integer, nullable=False)

    events: Mapped[List["Event"]] = relationship(back_populates="config")

############ AlchemySQL Database Structure ####################


############ Acessing the Database ####################
# Create an SQLite database (or connect to it if it already exists)
database_url = 'sqlite:///events.db'
engine = create_engine(database_url, echo=True)

# Create a configured "Session" class
Session = sessionmaker(bind=engine)
session = Session()

config_query = session.query(File_config).all()
config_data = [{'id': c.id, 'code_base': c.code_base, 'version': c.version, 'behaviour': c.behaviour, 'trial_num': c.trial_num} for c in config_query]
config_df = pd.DataFrame(config_data)
# print('config_data:', config_df)
# print([{'label': f"{row['code_base']} {row['version']} {row['behaviour']} {row['trial_num']}", 'value': row['id']} for _, row in config_df.iterrows()])

# Close the session
session.close()

# ### Filter the data based on the selected configuration and date range (TESTING)
# filtered_df = events_df[(events_df['config_id'] == config_data[0]['id'])]
# ### get in format required for plotting: [time, trace]
# filtered_df = filtered_df[['time', 'trace']]

############ Acessing the Database ####################

##################################################
############ Dash Application ####################
##################################################

############ Dash Layout ####################  
# Initialize the Dash app
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])


# Define the layout of the app
app.layout = html.Div([
    html.H1("Event Data Dashboard"),
    html.Br(),
    html.H4("Select Experiment Config:"),

    dcc.Dropdown(
        id='config-dropdown',
        options=[{'label': f"{row['code_base']} {row['version']} {row['behaviour']} {row['trial_num']}", 'value': row['id']} for _, row in config_df.iterrows()],
        value=config_df['id'].iloc[22],
        clearable=False,
        style={'width': COMP_WIDTH}
        ),

    html.Br(),

    html.H4("Select Range (Event Trace):"),
    dcc.RangeSlider(0, 100, 1, 
                    value=[0,20], 
                    marks=None, 
                    pushable=20,
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                        "style": {"color": "LightSteelBlue", "fontSize": "16px",},
                    },
                    id='range-slider',
                    ),

    html.Br(),
    html.H4("Select Plotting Parameters:"),
    dcc.Checklist(
        id ='addons',
        options = ['with_time', 'x_ticks', 'labels', 'thresholds', 'variables'],
        value = ['thresholds']
        # inline=True
    ),

    html.Br(),
    html.H4("Select Thresholds for EI:"),
    dcc.RadioItems(['Single-MinMax', 'Multi-MinMax',], 'Single-MinMax', inline=True, id='addons_thresholds', inputStyle={"margin-left": "20px", "margin-right": "5px"}),

    html.Br(),
    html.H4("Select Detection Model to Vizualize Predictions:"),
    dcc.Dropdown(['st_predictions', 'ei_predictions', 'ei_multithresh', 'st10_predictions', 'lstm_predictions', 'gru_predictions', 'forecaster_predictions', 'clustering_predictions', 'diag_AP2'], None, id='detection_model', style={'width': COMP_WIDTH}),

    html.Br(),
    html.H4("Select Subset of Predictions:"),
    dcc.Dropdown(['all_predict', 'tp_predict', 'fp_predict'], 'all_predict', id='detection_subset', style={'width': COMP_WIDTH}),

    html.Br(),
    html.H4("Merge with diff_val (seconds):"),
    dcc.Dropdown(['0', '1', '2', '5', '10', '15', '20', '25', '30', '35', '50', '100', '150'], '5', id='diff_val', style={'width': COMP_WIDTH}),

    html.Br(),
    html.H4("Window (only for ST):"),
    dcc.Dropdown(['10', '20', '30', '50', '80', '500'], '30', id='window', style={'width': COMP_WIDTH}),

    html.Br(),
    html.H4("Anomaly Seperation Method (only for diag_AP2)"),
    dcc.Dropdown(['1', '2', '3', '4'], '1', id='anomaly_sep', style={'width': COMP_WIDTH}),

    html.Br(),
    dcc.Loading(
            [dcc.Graph(id='time-series-plot')],
            overlay_style={"visibility":"visible", "filter": "blur(2px)"},
            type="circle",),
    
    # html.H4("Select Range (Exe Interval):"),
    # dcc.RangeSlider(0, 100, 1, value=[0,10], id='range-exeint'),

    html.Br(),

    html.Div(id="tab-content1", className="p-4", children=['Execution Interval Graphs'],
             style={'fontSize': '30px',}
            ),
                
    dcc.Loading(
            [html.Div(id="tab-content", className="p-4"),],
            overlay_style={"visibility":"visible", "filter": "blur(2px)"},
            type="circle", style={'padding': '20px'}),
    
], style={'width': '60%', 'margin': 'auto', 'padding': '20px', 'fontsize': '20px'})

############ Dash Layout ####################


############# Dash Functionality ####################
# Define the callback to update the graph
@app.callback(
    Output('time-series-plot', 'figure'),
    [Input('config-dropdown', 'value'), 
     Input('range-slider', 'value'), 
     Input('addons', 'value'),
     Input('detection_model', 'value'),
     Input('detection_subset', 'value'),
     Input('diff_val', 'value'),
     Input('anomaly_sep', 'value'),
     Input('window', 'value')],
)
def update_graph(selected_config_id, selected_range, addons_flags, detection_model, detection_subset, diff_val, anomaly_sep, window):
    session = Session()
    events_query = session.query(Event).filter_by(file_number=selected_config_id).all()
    # print('events_query:', len(events_query))
    events_data = [{'time': e.timestamp, 'trace': e.name,'config_id': e.file_number} for e in events_query]
    events_df = pd.DataFrame(events_data)
    session.close()

    ### get in format required for plotting: [time, trace]
    filtered_df = events_df[['time', 'trace']]
    # print('filtered_df time:', filtered_df['time'])
    timestamps = filtered_df['time']
    df_length = filtered_df.shape[0]
    start_index = int(selected_range[0] * df_length / 100)
    end_index = int(selected_range[1] * df_length / 100)

    # print('congif_df:', config_data)
    # print('selected_config_id:', selected_config_id)
    # print('codebase:', config_df.loc[config_df['id'] == selected_config_id, 'code_base'].iloc[0])

    CODE = config_df.loc[config_df['id'] == selected_config_id, 'code_base'].iloc[0]
    VERSION = config_df.loc[config_df['id'] == selected_config_id, 'version'].iloc[0]
    BEHAVIOUR = config_df.loc[config_df['id'] == selected_config_id, 'behaviour'].iloc[0]
    TRIAL = config_df.loc[config_df['id'] == selected_config_id, 'trial_num'].iloc[0]
    # print('CODE:', CODE, 'VERSION:', VERSION, 'BEHAVIOUR:', BEHAVIOUR, 'TRIAL:', TRIAL)
    if diff_val is not None:
        diff_val = int(diff_val)
        # print('diff_val:', diff_val)

    if anomaly_sep is not None:
        anomaly_sep = int(anomaly_sep)
        # print('anomaly_sep:', anomaly_sep)

    varlist_path = [f'../trace_data/{CODE}/single_thread/version_{VERSION}/{BEHAVIOUR}/varlist_trial{TRIAL}.json']
    label_path = [f'../trace_data/{CODE}/single_thread/version_{VERSION}/{BEHAVIOUR}/labels/trace_trial{TRIAL}_labels.json']
    
    predictions_path_ei = [f'../trace_data/{CODE}/single_thread/version_{VERSION}/{BEHAVIOUR}/ei_detections/trace_trial{TRIAL}_ei_detections_{diff_val}.json']
    predictions_path_ei_tp = [f'../trace_data/{CODE}/single_thread/version_{VERSION}/{BEHAVIOUR}/ei_detections/trace_trial{TRIAL}_tp_ei_detections_{diff_val}.json']
    predictions_path_ei_fp = [f'../trace_data/{CODE}/single_thread/version_{VERSION}/{BEHAVIOUR}/ei_detections/trace_trial{TRIAL}_fp_ei_detections_{diff_val}.json']

    predictions_path_ei_multithresh = [f'../trace_data/{CODE}/single_thread/version_{VERSION}/{BEHAVIOUR}/ei_multithresh_detections/trace_trial{TRIAL}_ei_multithresh_detections_{diff_val}.json']
    predictions_path_ei_multithresh_tp = [f'../trace_data/{CODE}/single_thread/version_{VERSION}/{BEHAVIOUR}/ei_multithresh_detections/trace_trial{TRIAL}_tp_ei_multithresh_detections_{diff_val}.json']
    predictions_path_ei_multithresh_fp = [f'../trace_data/{CODE}/single_thread/version_{VERSION}/{BEHAVIOUR}/ei_multithresh_detections/trace_trial{TRIAL}_fp_ei_multithresh_detections_{diff_val}.json']
    
    predictions_path_st = [f'../trace_data/{CODE}/single_thread/version_{VERSION}/{BEHAVIOUR}/st_detections/trace_trial{TRIAL}_st_detections_{diff_val}.json']
    predictions_path_st_tp = [f'../trace_data/{CODE}/single_thread/version_{VERSION}/{BEHAVIOUR}/st_detections/trace_trial{TRIAL}_tp_st_detections_{diff_val}.json']
    predictions_path_st_fp = [f'../trace_data/{CODE}/single_thread/version_{VERSION}/{BEHAVIOUR}/st_detections/trace_trial{TRIAL}_fp_st_detections_{diff_val}.json']
    # print('var_list_path_ET', varlist_path)

    predictions_path_st10 = [f'../trace_data/{CODE}/single_thread/version_{VERSION}/{BEHAVIOUR}/st{window}_detections/trace_trial{TRIAL}_st{window}_detections_{diff_val}.json']
    predictions_path_st10_tp = [f'../trace_data/{CODE}/single_thread/version_{VERSION}/{BEHAVIOUR}/st{window}_detections/trace_trial{TRIAL}_tp_st{window}_detections_{diff_val}.json']
    predictions_path_st10_fp = [f'../trace_data/{CODE}/single_thread/version_{VERSION}/{BEHAVIOUR}/st{window}_detections/trace_trial{TRIAL}_fp_st{window}_detections_{diff_val}.json']

    predictions_path_lstm = [f'../trace_data/{CODE}/single_thread/version_{VERSION}/{BEHAVIOUR}/lstm_detections/trace_trial{TRIAL}_lstm_detections_{diff_val}.json']
    predictions_path_lstm_tp = [f'../trace_data/{CODE}/single_thread/version_{VERSION}/{BEHAVIOUR}/lstm_detections/trace_trial{TRIAL}_tp_lstm_detections_{diff_val}.json']
    predictions_path_lstm_fp = [f'../trace_data/{CODE}/single_thread/version_{VERSION}/{BEHAVIOUR}/lstm_detections/trace_trial{TRIAL}_fp_lstm_detections_{diff_val}.json']

    predictions_path_gru = [f'../trace_data/{CODE}/single_thread/version_{VERSION}/{BEHAVIOUR}/gru_detections/trace_trial{TRIAL}_gru_detections_{diff_val}.json']
    predictions_path_gru_tp = [f'../trace_data/{CODE}/single_thread/version_{VERSION}/{BEHAVIOUR}/gru_detections/trace_trial{TRIAL}_tp_gru_detections_{diff_val}.json']
    predictions_path_gru_fp = [f'../trace_data/{CODE}/single_thread/version_{VERSION}/{BEHAVIOUR}/gru_detections/trace_trial{TRIAL}_fp_gru_detections_{diff_val}.json']

    predictions_path_forecaster = [f'../trace_data/{CODE}/single_thread/version_{VERSION}/{BEHAVIOUR}/forecaster_detections/trace_trial{TRIAL}_forecaster_detections_{diff_val}.json']
    predictions_path_forecaster_tp = [f'../trace_data/{CODE}/single_thread/version_{VERSION}/{BEHAVIOUR}/forecaster_detections/trace_trial{TRIAL}_tp_forecaster_detections_{diff_val}.json']
    predictions_path_forecaster_fp = [f'../trace_data/{CODE}/single_thread/version_{VERSION}/{BEHAVIOUR}/forecaster_detections/trace_trial{TRIAL}_fp_forecaster_detections_{diff_val}.json']

    predictions_path_diag = [f'../trace_data/{CODE}/single_thread/version_{VERSION}/{BEHAVIOUR}/diag{anomaly_sep}_detections/trace_trial{TRIAL}_diag{anomaly_sep}_detections_{diff_val}.json']
    predictions_path_diag_tp = [f'../trace_data/{CODE}/single_thread/version_{VERSION}/{BEHAVIOUR}/diag{anomaly_sep}_detections/trace_trial{TRIAL}_tp_diag{anomaly_sep}_detections_{diff_val}.json']
    predictions_path_diag_fp = [f'../trace_data/{CODE}/single_thread/version_{VERSION}/{BEHAVIOUR}/diag{anomaly_sep}_detections/trace_trial{TRIAL}_fp_diag{anomaly_sep}_detections_{diff_val}.json']


    ############# check varlist is consistent ############
    ############# only for version 3 ######################

    to_number = read_json(varlist_path[0])
    from_number = mapint2var(to_number)

    ############ Get variable list ######################
    sorted_keys = list(from_number.keys())
    sorted_keys.sort()
    var_list = [from_number[key] for key in sorted_keys]   ### get the variable list
    # print(var_list)

    ############# get the selected range ###############
    # print('selected range:', selected_range)
    # print('calculated indices:', start_index, end_index)
    selected_df = filtered_df.iloc[start_index:end_index]
    # print('selected_df:', selected_df)
    # print('selected_df shape:', selected_df.shape)

    ############## get ground truths ####################
    # print('adding labels:', addons_flags)
    labels = None
    predictions = None
    plot_time = False
    plot_x_ticks = False
    y_ticks = sorted_keys
    if addons_flags is not None:
        if 'labels' in addons_flags:
            ### check if label file exists
            if os.path.exists(label_path[0]):
                labels = prepare_labels(label_path)   ### need input as a list
                # labels = labels[start_index:end_index]
                # print('labels:', labels)
            else:
                labels = None
                print('Label file does not exist')

        if 'with_time' in addons_flags:
            plot_time = True

        if 'x_ticks' in addons_flags:
            plot_x_ticks = True

        if 'variables' in addons_flags:
            y_ticks = var_list

        ############## get predictions ####################
    if detection_model is not None:
        if 'ei_predictions' in detection_model:
            if 'all_predict' in detection_subset:
                if os.path.exists(predictions_path_ei[0]):
                    predictions = prepare_detections(predictions_path_ei, timestamps)
                else:
                    print('Prediction file does not exist')
            elif 'tp_predict' in detection_subset:
                if os.path.exists(predictions_path_ei_tp[0]):
                    predictions = prepare_detections(predictions_path_ei_tp, timestamps)
                else:
                    print('Prediction file does not exist')
            elif 'fp_predict' in detection_subset:
                if os.path.exists(predictions_path_ei_fp[0]):
                    predictions = prepare_detections(predictions_path_ei_fp, timestamps)
                else:
                    print('Prediction file does not exist')
        elif 'ei_multithresh' in detection_model:
            if 'all_predict' in detection_subset:
                if os.path.exists(predictions_path_ei_multithresh[0]):
                    predictions = prepare_detections(predictions_path_ei_multithresh, timestamps)
                else:
                    print('Prediction file does not exist')
            elif 'tp_predict' in detection_subset:
                if os.path.exists(predictions_path_ei_multithresh_tp[0]):
                    predictions = prepare_detections(predictions_path_ei_multithresh_tp, timestamps)
                else:
                    print('Prediction file does not exist')
            elif 'fp_predict' in detection_subset:
                if os.path.exists(predictions_path_ei_multithresh_fp[0]):
                    predictions = prepare_detections(predictions_path_ei_multithresh_fp, timestamps)
                else:
                    print('Prediction file does not exist')
        elif 'st_predictions' in detection_model:
            if 'all_predict' in detection_subset:
                if os.path.exists(predictions_path_st[0]):
                    predictions = prepare_detections(predictions_path_st, timestamps)
                else:
                    print('Prediction file does not exist')
            elif 'tp_predict' in detection_subset:
                if os.path.exists(predictions_path_st_tp[0]):
                    predictions = prepare_detections(predictions_path_st_tp, timestamps)
                else:
                    print('Prediction file does not exist')
            elif 'fp_predict' in detection_subset:
                if os.path.exists(predictions_path_st_fp[0]):
                    predictions = prepare_detections(predictions_path_st_fp, timestamps)
                else:
                    print('Prediction file does not exist')
        elif 'st10_predictions' in detection_model:
            if 'all_predict' in detection_subset:
                if os.path.exists(predictions_path_st10[0]):
                    predictions = prepare_detections(predictions_path_st10, timestamps)
                else:
                    print('Prediction file does not exist')
            elif 'tp_predict' in detection_subset:
                if os.path.exists(predictions_path_st10_tp[0]):
                    predictions = prepare_detections(predictions_path_st10_tp, timestamps)
                else:
                    print('Prediction file does not exist')
            elif 'fp_predict' in detection_subset:
                if os.path.exists(predictions_path_st10_fp[0]):
                    predictions = prepare_detections(predictions_path_st10_fp, timestamps)
                else:
                    print('Prediction file does not exist')
        elif'forecaster_predictions' in detection_model:
            if 'all_predict' in detection_subset:
                if os.path.exists(predictions_path_forecaster[0]):
                    predictions = prepare_detections(predictions_path_forecaster, timestamps)
                else:
                    print('Prediction file does not exist')
            elif 'tp_predict' in detection_subset:
                if os.path.exists(predictions_path_forecaster_tp[0]):
                    predictions = prepare_detections(predictions_path_forecaster_tp, timestamps)
                else:
                    print('Prediction file does not exist')
            elif 'fp_predict' in detection_subset:
                if os.path.exists(predictions_path_forecaster_fp[0]):
                    predictions = prepare_detections(predictions_path_forecaster_fp, timestamps)
                else:
                    print('Prediction file does not exist')
        elif 'lstm_predictions' in detection_model:
            if 'all_predict' in detection_subset:
                if os.path.exists(predictions_path_lstm[0]):
                    predictions = prepare_detections(predictions_path_lstm, timestamps)
                else:
                    print('Prediction file does not exist')
            elif 'tp_predict' in detection_subset:
                if os.path.exists(predictions_path_lstm_tp[0]):
                    predictions = prepare_detections(predictions_path_lstm_tp, timestamps)
                else:
                    print('Prediction file does not exist')
            elif 'fp_predict' in detection_subset:
                if os.path.exists(predictions_path_lstm_fp[0]):
                    predictions = prepare_detections(predictions_path_lstm_fp, timestamps)
                else:
                    print('Prediction file does not exist')
        elif 'gru_predictions' in detection_model:
            if 'all_predict' in detection_subset:
                if os.path.exists(predictions_path_gru[0]):
                    predictions = prepare_detections(predictions_path_gru, timestamps)
                else:
                    print('Prediction file does not exist')
            elif 'tp_predict' in detection_subset:
                if os.path.exists(predictions_path_gru_tp[0]):
                    predictions = prepare_detections(predictions_path_gru_tp, timestamps)
                else:
                    print('Prediction file does not exist')
            elif 'fp_predict' in detection_subset:
                if os.path.exists(predictions_path_gru_fp[0]):
                    predictions = prepare_detections(predictions_path_gru_fp, timestamps)
                else:
                    print('Prediction file does not exist')
        elif 'diag_AP2' in detection_model:
            if 'all_predict' in detection_subset:
                if os.path.exists(predictions_path_diag[0]):
                    predictions = prepare_detections(predictions_path_diag, timestamps)
                else:
                    print('Prediction file does not exist')
            elif 'tp_predict' in detection_subset:
                if os.path.exists(predictions_path_diag_tp[0]):
                    predictions = prepare_detections(predictions_path_diag_tp, timestamps)
                else:
                    print('Prediction file does not exist')
            elif 'fp_predict' in detection_subset:
                if os.path.exists(predictions_path_diag_fp[0]):
                    predictions = prepare_detections(predictions_path_diag_fp, timestamps)
                else:
                    print('Prediction file does not exist')

        else:
            print('{} not found'.format(detection_model))

    fig = plot_single_trace(selected_df, 
                            y_ticks,
                            with_time=plot_time, 
                            is_xticks=plot_x_ticks, 
                            ground_truths=labels,
                            detections=predictions,
                            )
    return fig


# Define the callback to update the execution interval graph
@app.callback(
    Output('tab-content', 'children'),
    [Input('config-dropdown', 'value'), 
     Input('range-slider', 'value'), 
     Input('addons', 'value'), 
     Input('addons_thresholds', 'value')],  # Add the new input for thresholds
)
def update_exeint(selected_config_id, selected_range, addons_flags, addons_thresholds):
    session = Session()
    events_query = session.query(Event).filter_by(file_number=selected_config_id).all()
    # print('events_query:', len(events_query))
    events_data = [{'time': e.timestamp, 'trace': e.name,'config_id': e.file_number} for e in events_query]
    events_df = pd.DataFrame(events_data)
    session.close()

    ### get in format required for plotting: [time, trace]
    filtered_df = events_df[['time', 'trace']]
    df_length = filtered_df.shape[0]
    start_index = int(selected_range[0] * df_length / 100)
    end_index = int(selected_range[1] * df_length / 100)
    # print('selected range:', selected_range)
    # print('calculated indices:', start_index, end_index)
    selected_df = filtered_df.iloc[start_index:end_index]
    # print('selected_df:', selected_df)
    # print('selected_df shape:', selected_df.shape)
    # print('check if empty:', selected_df.empty)

    CODE = config_df.loc[config_df['id'] == selected_config_id, 'code_base'].iloc[0]
    VERSION = config_df.loc[config_df['id'] == selected_config_id, 'version'].iloc[0]
    BEHAVIOUR = config_df.loc[config_df['id'] == selected_config_id, 'behaviour'].iloc[0]
    TRIAL = config_df.loc[config_df['id'] == selected_config_id, 'trial_num'].iloc[0]
    # print('CODE:', CODE, 'VERSION:', VERSION, 'BEHAVIOUR:', BEHAVIOUR, 'TRIAL:', TRIAL)

    varlist_path = [f'../trace_data/{CODE}/single_thread/version_{VERSION}/{BEHAVIOUR}/varlist_trial{TRIAL}.json']
    label_path = [f'../trace_data/{CODE}/single_thread/version_{VERSION}/{BEHAVIOUR}/labels/trace_trial{TRIAL}_labels.json']
    
    predictions_path_ei = [f'../trace_data/{CODE}/single_thread/version_{VERSION}/{BEHAVIOUR}/ei_detections/trace_trial{TRIAL}_ei_detections.json']
    predictions_path_ei_tp = [f'../trace_data/{CODE}/single_thread/version_{VERSION}/{BEHAVIOUR}/ei_detections/trace_trial{TRIAL}_tp_ei_detections.json']
    predictions_path_ei_fp = [f'../trace_data/{CODE}/single_thread/version_{VERSION}/{BEHAVIOUR}/ei_detections/trace_trial{TRIAL}_fp_ei_detections.json']
    
    predictions_path_st = [f'../trace_data/{CODE}/single_thread/version_{VERSION}/{BEHAVIOUR}/st_detections/trace_trial{TRIAL}_st_detections.json']
    predictions_path_st_tp = [f'../trace_data/{CODE}/single_thread/version_{VERSION}/{BEHAVIOUR}/st_detections/trace_trial{TRIAL}_tp_st_detections.json']
    predictions_path_st_fp = [f'../trace_data/{CODE}/single_thread/version_{VERSION}/{BEHAVIOUR}/st_detections/trace_trial{TRIAL}_fp_st_detections.json']

    threshold_path = [f'../trace_data/{CODE}/single_thread/version_{VERSION}/faulty_data/thresholds.json']
    threshmmulti_path = [f'../trace_data/{CODE}/single_thread/version_{VERSION}/faulty_data/thresholds_multi.json']

    # print('var_list_path_ei', varlist_path)

    # if 'x_ticks' in addons_flags:
    #         plot_x_ticks = True

    ############# check varlist is consistent ############
    ############# only for version 3 ######################

    to_number = read_json(varlist_path[0])
    from_number = mapint2var(to_number)
        


    ############ Get variable list ######################
    sorted_keys = list(from_number.keys())
    sorted_keys.sort()
    var_list = [from_number[key] for key in sorted_keys]   ### get the variable list
    # print(var_list)

    var_timestamps = get_var_timestamps(df=selected_df, config=f'{CODE}_{VERSION}_{BEHAVIOUR}_{TRIAL}') 
    # print('var_timestamps:', var_timestamps)
    to_plot = preprocess_variable_plotting(var_timestamps, var_list, from_number)  
    # print('to_plot:', to_plot)

    ############ Get Thresholds ######################
    thresholds = None
    labels = None
    if addons_flags is not None:
        # if 'thresholds' in addons_flags:
        #     if os.path.exists(threshold_path[0]):
        #         thresholds = read_json(threshold_path[0])
        #     else:
        #         print('Threshold file does not exist')
        
        
        if 'labels' in addons_flags:
            ### check if label file exists
            if os.path.exists(label_path[0]):
                labels = prepare_labels(label_path)   ### need input as a list
                # labels = labels[start_index:end_index]
                # print('labels:', labels)
            else:
                labels = None
                print('Label file does not exist')
            
    if addons_thresholds is not None:
        if addons_thresholds == 'Single-MinMax':
            if os.path.exists(threshold_path[0]):
                thresholds = read_json(threshold_path[0])
            else:
                print('Threshold file does not exist')
        elif addons_thresholds == 'Multi-MinMax':
            if os.path.exists(threshmmulti_path[0]):
                thresholds = read_json(threshmmulti_path[0])
            else:
                print('Threshold file does not exist')
        else:
            print('Thresholds not found')

    plot_list = plot_execution_interval_single(to_plot, 
                                               ground_truths= labels,
                                               is_xticks=False, 
                                               thresholds=thresholds,
                                               var2num=to_number)
    print('got plot_list')
    fig_list = []
    for fig in plot_list:
        fig_list.append(dcc.Graph(figure=fig))
    print('created objects')
    return fig_list
    # return dcc.Graph(figure=plot_list[0])

############# Dash Functionality ####################


######## Run the app ########
if __name__ == '__main__':
    app.run_server(debug=True)