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


# Create an SQLite database (or connect to it if it already exists)
database_url = 'sqlite:///events.db'
engine = create_engine(database_url, echo=True)

# Create a configured "Session" class
Session = sessionmaker(bind=engine)
session = Session()

config_query = session.query(File_config).all()
config_data = [{'id': c.id, 'code_base': c.code_base, 'version': c.version, 'behaviour': c.behaviour, 'trial_num': c.trial_num} for c in config_query]
config_df = pd.DataFrame(config_data)

# Close the session
session.close()

# ### Filter the data based on the selected configuration and date range (TESTING)
# filtered_df = events_df[(events_df['config_id'] == config_data[0]['id'])]
# ### get in format required for plotting: [time, trace]
# filtered_df = filtered_df[['time', 'trace']]

CODE = config_data[0]['code_base']
VERSION = config_data[0]['version']
BEHAVIOUR = config_data[0]['behaviour']
TRIAL = config_data[0]['trial_num']

varlist_path = [f'../trace_data/{CODE}/single_thread/version_{VERSION}/{BEHAVIOUR}/varlist_trial{TRIAL}.json']
print('var_list_path', varlist_path)

############# check varlist is consistent ############
############# only for version 3 ######################

if VERSION == 3:
    to_number = is_consistent(varlist_path)

    if to_number != False:
        from_number = mapint2var(to_number)

############ Get variable list ######################
sorted_keys = list(from_number.keys())
sorted_keys.sort()
var_list = [from_number[key] for key in sorted_keys]   ### get the variable list
# print(var_list)

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Event Data Dashboard"),
    dcc.Dropdown(
        id='config-dropdown',
        options=[{'label': f"{row['code_base']} {row['version']} {row['behaviour']} {row['trial_num']}", 'value': row['id']} for _, row in config_df.iterrows()],
        value=config_df['id'].iloc[0],
        clearable=False
        ),
    html.Br(),
    html.H6("Select Event Trace Range to Display:"),
    dcc.RangeSlider(0, 100, 1, value=[0,20], id='range-slider'),
    html.Br(),
    dcc.Loading(
            [dcc.Graph(id='time-series-plot')],
            overlay_style={"visibility":"visible", "filter": "blur(2px)"},
            type="circle",)
    
])

# Define the callback to update the graph
@app.callback(
    Output('time-series-plot', 'figure'),
    [Input('config-dropdown', 'value'), Input('range-slider', 'value')]
)
def update_graph(selected_config_id, selected_range):
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
    fig = plot_single_trace(selected_df, sorted_keys, with_time=False, is_xticks=True)
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)