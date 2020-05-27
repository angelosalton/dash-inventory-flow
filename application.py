# Angelo Salton <angelo.salton@slcagricola.com.br>

import base64
import io
import random
from itertools import permutations

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from flask import Flask

server = Flask(__name__)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE])

# load map
net = pickle.load(open('map.pkl', 'rb'))

# layout
navbar = dbc.Navbar([
    dbc.Row([
        dbc.NavbarBrand('dash-optim', className='ml-2'),
        html.A([
            html.Img(
                src='https://img.shields.io/github/forks/angelosalton/dash-optim?style=social', height='20px')
        ],
            href='https://github.com/angelosalton/dash-optim/fork')
    ],
        align='center',
        no_gutters=True)
],
    color='dark',
    dark=True
)

sidebar = dbc.Card([
    html.H4('Simulation parameters'),
    # location
    html.P('Locations', style={'margin-bottom': '0px'}),
    dbc.Input(id='input-locations', type='number', min=2,
              max=20, value=4, style={'margin-bottom': '20px'}),
    # item
    html.P('Items', style={'margin-bottom': '0px'}),
    dbc.Input(id='input-items', type='number', min=1, max=5,
              value=3, style={'margin-bottom': '20px'}),
    # run
    dbc.Button('Update', id='btn-update')
], style={'padding': '2em 2em'})

app.layout = dbc.Container([
    navbar,
    # data stores
    dcc.Store(id='data-store', storage_type='session'),
    # top alert
    dbc.Alert("Welcome!", color="success", fade=True),
    # main
    dbc.Row([
        dcc.Markdown('''
        Welcome! This is a simulation of the multicommodity flow problem: there are a number of locations/warehouses with supply and demand of a number of non-substitutible goods. A solution for the problem is a program of transfers that minimize excess supply/demand of each item.
        '''),
        # sidebar
        dbc.Col([
            dbc.Row([
                sidebar
            ])
        ], md=2),
        # main
        dbc.Col([
            dbc.Row([
                dcc.Markdown('### Hello!')
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Markdown('#### Solution'),
                    html.P('Item to display', style={'margin-bottom': '0px'}),
                    dcc.Dropdown(id='item-drop-div'),
                    html.Div(id='plot-solution')
                ]),
                dbc.Col([
                    dcc.Markdown('#### Data'),
                    html.Div(id='data-table')
                ])
            ])
        ], md=10, style={'padding-left': '30px'})
    ])]
)

# Functions


def fig_to_uri(in_fig: mpl.figure.Figure, close_all=True, **save_args) -> str:
    '''
    Generate .png from matplotlib figure.
    '''
    buffer = io.BytesIO()
    in_fig.savefig(buffer, format='png', **save_args)

    # close previous figures
    # if close_all:
    #     in_fig.clf()
    # plt.close('all')

    buffer.seek(0)  # rewind file
    encoded = base64.b64encode(buffer.getbuffer()).decode(
        "ascii").replace("\n", "")

    return "data:image/png;base64,{}".format(encoded)


# Callbacks
@app.callback(
    Output('data-store', 'data'),
    [Input('btn-update', 'n_clicks')],
    [State('input-locations', 'value'),
     State('input-items', 'value'),
     State('data-store', 'data')])
def dataset_gen(n_clicks, locations, items, df):
    '''
    Generate dataset for dash-logit.
    '''
    df = pd.DataFrame()

    df['location'] = [chr(x) for x in range(65, 65+locations)]*items
    df['item'] = sorted([*range(1, items+1)]*locations)
    df['supply'] = random.sample(range(0, 200, 10), locations*items)
    df['demand'] = random.sample(range(0, 200, 10), locations*items)
    df['net_demand'] = df['demand'] - df['supply']

    return df.to_json(orient='records')

@app.callback(
    Output('data-table', 'children'),
    [Input('data-store', 'modified_timestamp'),
     Input('btn-update', 'n_clicks')],
    [State('data-store', 'data')])
def print_datatable(ts, clicks, df_json):
    '''
    Print a data table.
    '''
    if df_json is None:
        raise PreventUpdate

    dff = pd.read_json(df_json, orient='records')

    return dt.DataTable(
        data=dff.to_dict(orient='records'),
        columns=[{"name": i, "id": i} for i in dff.columns],
        editable=True,
        style_cell={
            'fontSize': 12,
            'font-family': 'Roboto',
            'maxWidth': '40px'
        }
    )


# item dropdown
@app.callback(
    Output('item-drop-div', 'options'),
    [Input('data-store', 'data')])
def update_dropdown(df_json):

    if df_json is None:
        raise PreventUpdate

    df = pd.read_json(df_json, orient='records')
    opts = [{'label': str(i), 'value': i}
            for i in df['item'].unique().tolist()]

    return opts


# call network generator
@app.callback(
    Output('plot-solution', 'children'),
    [Input('btn-update', 'n_clicks'),
     Input('item-drop-div', 'value')],
    [State('data-store', 'data'),
     State('distance-store', 'data')])
def network_gen(clicks, item, df_json, distances):
    '''
    Generates a networkx graph from data.
    '''
    if df_json is None or distances is None:
        raise PreventUpdate

    # subset data
    df = pd.read_json(df_json, orient='records')

    list_items = df['item'].unique().tolist()

    try:
        dft = df[df['item'] == list_items[item-1]]
    except:
        raise PreventUpdate

    # calculate excess supply/demand
    exc_dem = dft['net_demand'].sum() if dft['net_demand'].sum(
    ) < 0 else -dft['net_demand'].sum()

    # add dummy node
    dummy_node = {
        'location': 'dummy',
        'item': item,
        'supply': 0 if exc_dem > 0 else exc_dem,
        'demand': exc_dem if exc_dem > 0 else 0,
        'net_demand': -exc_dem if exc_dem > 0 else exc_dem,
    }

    dft = dft.append(dummy_node, ignore_index=True)
    dft.fillna(0, inplace=True)

    # move data to dict
    records = dft[['location', 'net_demand']].to_dict('records')

    # create graph
    Gr = nx.DiGraph()

    # get unique locations
    locs = dft['location'].unique()

    # add nodes
    [Gr.add_node(item['location'], demand=item['net_demand'])
        for item in records]

    # edge weights
    node_weights = distances #{j: {i: 1 for i in locs} for j in locs}

    # add edges
    [Gr.add_edge(i, j, weight=1, capacity=100)
        for (i, j) in permutations(locs, 2)]

    def graph_gen(gr=Gr):
        '''
        Calculate and plot optimal flows.

        :param gr: A networkx network.
        :returns: None
        '''
        try:
            _, flows = nx.network_simplex(gr)
        except nx.NetworkXUnfeasible:
            return html.Div([
                dcc.Markdown('### No feasible solution.')
            ])

        # drawing
        visible_edges = [(i, j)
                         for i, j in gr.edges if flows[i][j] > 0 and i != 'dummy']
        visible_nodes = [i for i in gr.nodes if i != 'dummy']
        visible_edgelabels = {(i[0], i[1]): flows[i[0]][i[1]]
                              for i in visible_edges}

        # set layout
        gr_layout = nx.layout.circular_layout(gr)

        fig1, ax1 = plt.subplots()
        nx.draw_networkx_nodes(
            gr, pos=gr_layout, nodelist=visible_nodes, node_size=1000, alpha=.9, ax=ax1, node_shape='^')
        nx.draw_networkx_labels(gr, pos=gr_layout, ax=ax1)
        nx.draw_networkx_edges(
            gr, pos=gr_layout, edgelist=visible_edges, min_target_margin=40, ax=ax1)
        nx.draw_networkx_edge_labels(
            gr, pos=gr_layout, edge_labels=visible_edgelabels, font_size=20, ax=ax1)

        layout = html.Div([
            html.Img(src=fig_to_uri(fig1), style={'max-width': '400px'})
        ])

        return layout

    # plot the graph
    return graph_gen()


if __name__ == "__main__":
    app.run_server(debug=True)
