import h5py
import numpy as np
import pandas as pd
from collections import OrderedDict
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from mni import create_mesh_data, default_colorscale
import plotly.graph_objs as go
import copy

# Import cortical thickness and connectivity data
with h5py.File("all_lnm_data_corrected.hdf5", 'r') as f:
    ages = f["ages"][()]
    ct_data = f["dkt/grey_matter_metrics/ct"][()]
    cortical_conn = f["dkt/connectivity_matrices/cortical_only"][()]

ct_data = np.reshape(ct_data, (ct_data.shape[0], ct_data.shape[1]))

# Remove data for regions not in DKT atlas (left6, right6)
ct_data = np.delete(ct_data, [5, 32 + 5], 1)

# Sort data by age
indices = ages.argsort()
ages.sort()
ct_data = ct_data[indices, :]
cortical_conn = cortical_conn[indices, :]

# Import DKT atlas data
with h5py.File("atlases.hdf5", 'r') as fa:
    dkt_ids_cortex = fa['gm/dkt/id_cortex'][()]
    region_names = fa['gm/dkt/name'][()]

regions = OrderedDict()
for i in np.arange(0, len(region_names)):
    if dkt_ids_cortex[i] != 'nan':
        id = int(dkt_ids_cortex[i])
        regions[id] = region_names[i]

# Import DKT conversions
dkt = pd.read_csv('data/dkt_conv.csv', sep='\t', names=['name', 'region'], header=None, skiprows=1)

# Define data options
data_options = ['Data & Trend Line', 'Data only', 'Trend Line only']


def estimate1param(rate, times, initial, step):
    output = np.zeros(times.shape[0])
    output[0] = initial
    i = 1
    for time in times[1:]:
        last = output[i - 1]

        deriv = rate * last
        output[i] = last + deriv * step
        i += 1

    return output


def estimate2param(iroc, eroc, times, initial, step):
    output = np.zeros((times.shape[0], len(initial)))
    output[0, :] = initial
    i = 1
    for time in times[1:]:
        last = output[i - 1, :]

        at = iroc * last

        m = np.tile(last, (len(initial), 1))
        np.fill_diagonal(m, 0.0)
        bt = np.sum(eroc * m.sum(-1))

        deriv = at + bt
        output[i, :] = last + deriv * step
        i += 1

    return output


cached_mesh = create_mesh_data("human_atlas", -1)

app = dash.Dash("CT Dashboard")

axis_template = {
    "showbackground": True,
    #   "backgroundcolor": "#141414",
    "gridcolor": "rgb(255, 255, 255)",
    "zerolinecolor": "rgb(255, 255, 255)",
}

plot_layout = {
    "title": "",
    "margin": {"t": 0, "b": 0, "l": 0, "r": 0},
    "font": {"size": 12, "color": "black"},
    "showlegend": False,
    # "plot_bgcolor": "#141414",
    # "paper_bgcolor": "#141414",
    "scene": {
        "xaxis": axis_template,
        "yaxis": axis_template,
        "zaxis": axis_template,
        "aspectratio": {"x": 1, "y": 1.2, "z": 1},
        "camera": {"eye": {"x": 1.25, "y": 1.25, "z": 1.25}},
        "annotations": [],
    },
}

app.layout = html.Div([
    dcc.Markdown('## Lifespan Cortical Thickness Data'),
    # Options for CT graphic
    html.Div(id='ct-graphic-options-div'),


    # CT graphic
    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    dcc.Markdown('Region:'),
                    dcc.Dropdown(
                        id='region',
                        options=[{'label': regions[i], 'value': regions[i]} for i in regions],
                        value=regions[25]
                    ),
                ],
                    style={'width': '30%', 'display': 'inline-block'}),

                html.Div([
                    dcc.Markdown('Plot Components:'),
                    dcc.Dropdown(
                        id='display_data',
                        options=[{'label': i, 'value': i} for i in data_options],
                        value=data_options[0]
                    ),
                ], style={'width': '22%', 'float': 'right', 'display': 'inline-block'}),
                ], style={'width': '60%'}),
            dcc.Graph(id='ct-graphic'),
        ],
            style={'width': '70%', 'display': 'inline-block'}),

        html.Div([
            dcc.Graph(
                id="brain-graph",
                figure={
                    "data": cached_mesh,
                    "layout": plot_layout,
                },
                config={"editable": True, "scrollZoom": False},
            )
        ],
            style={'width': '30%', 'float': 'right', 'display': 'inline-block'}),
    ]),

    html.Div([
    # CT simulator
    dcc.Markdown('## Single Parameter Model Simulation'),
    html.Div([
        html.Div([
            # CT simulator graphic
            dcc.Graph(id='ct-simulator'),
        ],
            style={'width': '75%', 'display': 'inline-block'}),

        html.Div([
            # dcc.Markdown('Options:'),
            html.P('Initial CT value:'),
            dcc.Input(id='initial_ct', placeholder='Enter initial CT value...',
                      type='text', value='3.83', style={'width': '40px'}),
            # html.Label('Rate of Change:'),
            html.P(),
            html.P('Rate of Change:'),
            dcc.Slider(
                id='roc',
                min=-0.05,
                max=0.05,
                step=0.005,
                value=-0.03,
                marks={
                    -0.05: 'Max Atrophy',
                    0: 'No Change',
                    0.05: 'Max Growth'
                },
            ),
            html.P('Ages:'),
            dcc.RangeSlider(
                id='age_slider',
                count=1,
                min=1,
                max=90,
                step=0.5,
                value=[7, 85],
                marks={0: '0', 50: '50', 90: '90'}
                # marks={i: '{}'.format(i) for i in np.arange(1, 85, 10)}
            )
        ], style={'width': '25%', 'float': 'right', 'display': 'inline-block'}),
    ])
    ]),
    html.Div([
    # CT 2 model simulator
    dcc.Markdown('## Two Parameter Model Simulation'),
    html.Div([
        html.Div([
            # CT simulator graphic
            dcc.Graph(id='ct-simulator2'),
        ],
            style={'width': '75%', 'display': 'inline-block'}),

        html.Div([
            # dcc.Markdown('Options:'),
            html.P('Initial CT value:'),
            dcc.Input(id='initial_ct2', placeholder='Enter initial CT value...',
                      type='text', value='3.83', style={'width': '40px'}),
            # html.Label('Rate of Change:'),
            html.P(),
            html.P('Internal Rate of Change:'),
            dcc.Slider(
                id='iroc',
                min=-0.05,
                max=0.05,
                step=0.005,
                value=-0.03,
                marks={
                    -0.05: 'Max Atrophy',
                    0: 'No Change',
                    0.05: 'Max Growth'
                },
            ),
            html.P(),
            html.P('External Rate of Change:'),
            dcc.Slider(
                id='eroc',
                min=-0.005,
                max=0.005,
                step=0.0005,
                value=0.003,
                marks={
                    -0.005: 'Max Atrophy',
                    0: 'No Change',
                    0.005: 'Max Growth'
                },
            ),
            html.P('Ages:'),
            dcc.RangeSlider(
                id='age_slider2',
                count=1,
                min=1,
                max=90,
                step=0.5,
                value=[7, 85],
                marks={0: '0', 50: '50', 90: '90'}
                # marks={i: '{}'.format(i) for i in np.arange(1, 85, 10)}
            )
        ], style={'width': '25%', 'float': 'right', 'display': 'inline-block'}),
    ])
    ])
])


def get_polynomial_trajectory(times, region_count, values, ages, degree):
    val = np.zeros((times.shape[0], region_count))
    for i in range(0, region_count):
        f = np.poly1d(np.polyfit(ages, values[:, i], degree))
        val[:, i] = f(times)

    return val


@app.callback(
    Output('ct-graphic', 'figure'),
    [Input(component_id='region', component_property='value'),
     Input(component_id='display_data', component_property='value')]
)
def update_ct_graphic(r, dd):
    if r is not None:
        id = -1
        # print(r)
        if dkt['name'].str.contains(r).any():
            try:
                id = dkt.index[dkt['name'] == r][0]
            except:
                id = -1

        outs = []
        # print(id)
        if dd != "Trend Line only":
            outs.append(go.Scatter(x=ages, y=ct_data[:, id], mode='markers', name='Data'))

        if dd != "Data only":
            traj = get_polynomial_trajectory(ages, 62, ct_data, ages, 3)

            outs.append(go.Scatter(x=ages, y=traj[:, id], name='Trend line'))

        if id >= 0:
            return {
                'data': outs,
                'layout': go.Layout(
                    xaxis={'title': 'Age (years)'},
                    yaxis={'title': '%s CT (cm)' % (r), 'range': [0, 4]},
                    margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                    legend={'x': 0, 'y': 1},
                    hovermode='closest')
            }
        else:
            return {'data': []}
    else:
        return {'data': []}


@app.callback(
    Output('brain-graph', 'figure'),
    [Input(component_id='region', component_property='value')]
)
def update_brain_graphic(r):
    if r is not None:
        # print(r)
        if dkt['name'].str.contains(r).any():
            try:
                region = dkt.loc[dkt['name'] == r]['region'].iloc[0]
            except:
                region = -1
        else:
            region = -1

        if region > -1:
            # Import views
            views = pd.read_csv('data/views.txt', delim_whitespace=True)

            x = views[views['region'] == region]['x'].iloc[0]
            y = views[views['region'] == region]['y'].iloc[0]
            z = views[views['region'] == region]['z'].iloc[0]

            # print(region)
            # print(x)
            # print(y)
            # print(z)
            # print(views.head())

            plot_layout['scene']['camera']['eye'] = {"x": x, "y": y, "z": z}
        else:
            plot_layout['scene']['camera']['eye'] = {"x": -1.25, "y": 1.25, "z": 1.25}

        temp_mesh = copy.deepcopy(cached_mesh)

        if region >= 0:
            temp_mesh[0]['intensity'][temp_mesh[0]['intensity'] != region] = 0

        return {
            'data': temp_mesh, #create_mesh_data("human_atlas", region),
            'layout': plot_layout
        }
    else:
        plot_layout['scene']['camera']['eye'] = {"x": -1.25, "y": 1.25, "z": 1.25}

        return {
            'data': cached_mesh, #create_mesh_data("human_atlas", region),
            'layout': plot_layout
        }


@app.callback(
    Output('ct-simulator', 'figure'),
    [Input(component_id='initial_ct', component_property='value'),
     Input(component_id='roc', component_property='value'),
     Input(component_id='age_slider', component_property='value')]
)
def update_ct_graphic(init_ct, roc, age_slider):
    ic = float(init_ct)
    mina = age_slider[0]
    maxa = age_slider[1]

    times = np.arange(mina, maxa, 0.5)
    estims = estimate1param(roc, times, ic, 0.5)

    outs = []
    outs.append(go.Scatter(x=times, y=estims, mode='markers', name='Data'))

    if roc < 0.0:
        range = [0, ic+1.0]
    else:
        range = [0, max(estims)+1.0]

    return {
        'data': outs,
        'layout': go.Layout(
            xaxis={'title': 'Age (years)'},
            yaxis={'title': 'Simulated CT (cm)', 'range': range},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend={'x': 0, 'y': 1},
            hovermode='closest')
    }


@app.callback(
    Output('ct-simulator2', 'figure'),
    [Input(component_id='initial_ct2', component_property='value'),
     Input(component_id='iroc', component_property='value'),
     Input(component_id='eroc', component_property='value'),
     Input(component_id='age_slider2', component_property='value')]
)
def update_ct_graphic2(init_ct, iroc, eroc, age_slider):
    ic = float(init_ct)
    mina = age_slider[0]
    maxa = age_slider[1]

    times = np.arange(mina, maxa, 0.5)
    estims = estimate2param(iroc, eroc, times, [ic, 2.78, 3.44], 0.5)

    outs = []
    outs.append(go.Scatter(x=times, y=estims[:, 0], mode='markers', name='Region 1'))
    outs.append(go.Scatter(x=times, y=estims[:, 1], mode='markers', name='Region 2'))
    outs.append(go.Scatter(x=times, y=estims[:, 2], mode='markers', name='Region 3'))

    if iroc < 0.0:
        range = [0, ic+1.0]
    else:
        range = [0, max(estims[:, 0])+1.0]

    return {
        'data': outs,
        'layout': go.Layout(
            xaxis={'title': 'Age (years)'},
            yaxis={'title': 'Simulated CT (cm)', 'range': range},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend={'x': 0, 'y': 1},
            hovermode='closest')
    }


if __name__ == '__main__':
    app.run_server(debug=True)
