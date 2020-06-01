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

# generate map
# boundaries - center Porto Alegre
# north, south, east, west = -30.0259, -30.0801, -51.1469, -51.2371

# mesh = osmnx.graph_from_bbox(north, south, east, west, network_type='drive')
# pickle.dump(mesh, open('map.pkl', 'wb'))

# load map
mesh = pickle.load(open('map.pkl', 'rb'))
mesh_nodes = [*mesh.nodes()]
mesh_edges = [*mesh.edges(data=True)]

# layout
navbar = dbc.Navbar([
    dbc.Row([
        dbc.NavbarBrand('dash-optim', className='ml-2'),
        html.A([
            html.Img(
                src='https://img.shields.io/github/forks/angelosalton/dash-inventory-flow?style=social', height='20px')
        ],
            href='https://github.com/angelosalton/dash-inventory-flow/fork')
    ],
        align='center',
        no_gutters=True)
],
    color='dark',
    dark=True
)

parameters = dbc.Row([
    html.H4('Simulation parameters'),
    # location
    html.P('Locations', style={'margin-bottom': '0px'}),
    dbc.Input(id='input-locations', type='number', min=2,
              max=20, value=4, style={'margin-bottom': '20px'}),
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
        # parameters
        dbc.Col([
            dbc.Row([
                parameters
            ])
        ], md=2),
        # main
        dbc.Col([
            dbc.Row([
                dcc.Markdown('### Hello!')
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Markdown('#### Map'),
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
     State('data-store', 'data')])
def dataset_gen(clicks, n_nodes, df_json):
    '''
    Generate dataset for dash-optim.
    '''
    df = pd.DataFrame()
    
    # get random nodes
    rnodes = random.choices(mesh_nodes, k=n_nodes)

    df['location'] = rnodes
    df['supply'] = random.sample(range(0, 200, 10), n_nodes)
    df['demand'] = random.sample(range(0, 200, 10), n_nodes)
    df['net_demand'] = df['demand'] - df['supply']
    
    dfr = df.append(dummy_node, ignore_index=True)

    return dfr.to_json(orient='records')


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

# call network generator
@app.callback(
    Output('plot-solution', 'children'),
    [Input('btn-update', 'n_clicks')],
    [State('data-store', 'data')])
def model_solution(clicks, df_json):
    '''
    Outputs the solution of model.
    '''
    if df_json is None:
        raise PreventUpdate

    dff = pd.read_json(df_json, orient='records')

    # get number of nodes from data
    n_nodes = dff['location'].unique().count()

    # get the randomized nodes
    rnodes = dff['location'].tolist()

    # calculate shortest paths
    paths = {i: {j: nx.shortest_path(mesh, i, j, weight='length') for j in nodelist} for i in nodelist}

    def simplified_graph(graph, nodelist, pathlist):
        '''
        Return a simplified graph from OSMnx with shortest distances.
        '''
        def calc_distance(graph, path):
            '''
            Get total distance of a path in the network.
            '''
            dist = round(sum(ox.utils_graph.get_route_edge_attributes(graph, path, 'length'))/1000., 2)
            return dist
        
        distances = {i: {j: calc_distance(graph, pathlist[i][j]) for j in nodelist} for i in nodelist}

        # create simplified graph
        subgraph = nx.DiGraph()
        
        # add nodes
        [subgraph.add_node(n) for n in nodelist]

        # add edges
        [subgraph.add_edge(i, j, weight=distances[i][j]) for (i,j) in permutations(nodelist,2)]
        
        return subgraph

    # get random nodes
    rnodes = random.choices(mesh_nodes, k=n_nodes)
    
    # create simplified graph
    sgraph = simplified_graph(mesh, rnodes, paths)

    # calculate excess supply/demand
    exc_dem = df['net_demand'].sum()

    # add dummy node
    sgraph.add_node('dummy')
    [sgraph.add_edge('dummy', n, length=0.1) for n in rnodes]

    # adding demand
    [sgraph.add_node(n, demand=int(dftest[dftest['location']==n]['net_demand'])) for n in rnodes]
    sgraph.add_node('dummy', demand=-exc_dem)

    try:
        _, flows = nx.network_simplex(sgraph)
    except nx.NetworkXUnfeasible:
        return html.Div([
            dcc.Markdown('Solution not found.')
        ])

    # nodes positions
    rnodes_attr = {n: mesh.nodes[n] for n in rnodes}

    # initialize figure
    fig = ox.plot_route_folium(mesh, paths[rnodes[0]][rnodes[4]], popup_attribute='name', route_opacity=.7)

    # add nodes
    [ox.folium.folium.Marker([rnodes_attr[n]['y'], rnodes_attr[n]['x']], tooltip=str(n)).add_to(fig) for n in rnodes]

    # add routes
    ox.plot_route_folium(mesh, paths[rnodes[0]][rnodes[3]], route_map=fig, popup_attribute='name', route_opacity=.7)

    # show figure
    return html.Div([
        dcc.Markdown('Ok!')
    ])
    

if __name__ == "__main__":
    app.run_server(debug=True)
