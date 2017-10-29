# -*- coding: utf-8 -*-

"""
Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np
import dash, os, json
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State, Event
from plotly.tools import make_subplots
from flask import Flask, send_from_directory
from stochopy import BenchmarkFunction, Evolutionary


MAX_SEED = 999999
FUNCOPT = (
    "Ackley",
    "Griewank",
    "Quartic",
    "Quartic noise",
    "Rastrigin",
    "Rosenbrock",
    "Sphere",
    "Styblinski-Tang",
    )
SOLVOPT = (
    "CPSO",
    "PSO",
    "DE",
    "CMAES",
    )
DESTOPT = (
    "rand1",
    "rand2",
    "best1",
    "best2",
    )
func_options = [ dict(label = func, value = "_".join(func.split()).lower())
                    for func in FUNCOPT ]
solver_options = [ dict(label = solver, value = solver.lower()) for solver in SOLVOPT ]
destrat_options = [ dict(label = strat, value = strat.lower()) for strat in DESTOPT ]
delay = 100
bgcolor = "#FFFFFF"
welcome_text = "[StochOPy](https://github.com/keurfonluu/StochOPy) (**STOCH**astic **OP**timization for **PY**thon) is a package that provides user-friendly routines to sample or optimize objective functions with the most popular algorithms." \
                + "This WebApp allows the users to see how popular stochastic algorithms perform on different benchmark functions."


server = Flask(__name__)
server.secret_key = os.environ.get("secret_key", "secret")
app = dash.Dash(__name__, server = server)
app.title = "StochOPy WebApp"
app.config.supress_callback_exceptions = True
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})
#app.css.config.serve_locally = True
#app.scripts.config.serve_locally = True
app.layout = html.Div([
    # CSS
#    html.Link(rel = "stylesheet", href = "/static/style.css"),
        
    # Header
    html.Div([
        html.H1(
            "StochOPy WebApp",
            style = dict(textAlign = "center"),
            ),
        dcc.Markdown(
            welcome_text,
            ),
        ],
        style = dict(
            width = "100%",
            ),
        ),
    
    # Separator
    html.Hr(
        style = dict(
            marginTop = "0",
            marginBottom = "0",
            marginLeft = "10",
            marginRight = "10",
            ),
        ),
                
    # GUI
    html.Div([
        # Parameter panel
        html.Div([
            # Parameters
            html.Div([
                # Function
                html.Label("Function"),
                dcc.Dropdown(
                    id = "function",
                    options = func_options,
                    value = "rosenbrock",
                    multi = False,
                    ),
                html.Br(),
                
                html.Div([
                    # Maximum number of iterations
                    html.Label("# of iterations", style = dict(display = "block")),
                    dcc.Input(
                        id = "max_iter",
                        value = 200,
                        min = 0,
                        type = "number",
                        style = dict(
                            width = "100%",
                            display = "inline-block",
                            ),
                        ),
                    
                    # Population size
                    html.Label("Population size", style = dict(display = "block")),
                    dcc.Input(
                        id = "population_size",
                        value = 10,
                        min = 2,
                        type = "number",
                        style = dict(
                            width = "100%",
                            display = "inline-block",
                            ),
                        ),
                    ],
                    style = dict(
                        columnCount = 2,
                        ),
                    ),
                
                # Seed
                html.Div([
                    html.Div([
                        dcc.Checklist(
                            id = "seed_check",
                            options = [ dict(label = "Fix seed", value = "fix_seed") ],
                            values = [ "fix_seed" ],
                            ),
                        ],
                        style = dict(
                            width = "40%",
                            display = "inline-block",
                            verticalAlign = "middle",
                            ),
                        ),
                    dcc.Input(
                        id = "seed",
                        value = 42,
                        min = 0,
                        type = "number",
                        style = dict(
                            width = "60%",
                            display = "inline-block",
                            ),
                        ),
                    ],
                    style = dict(
                        display = "inline-block",
                        marginTop = "10",
                        ),
                    ),
                
                ],
                ),
            
            # Separator
            html.Hr(style = dict(margin = "15")),
                        
            # Solver
            html.Div([
                html.Label("Solver", style = dict(display = "block")),
                dcc.Dropdown(
                    id = "solver",
                    options = solver_options,
                    value = "cpso",
                    multi = False,
                    ),
                html.Br(),
                
                # PSO parameters
                html.Div([
                    # Inertial weight
                    html.Div([
                        html.Label("Inertia weight", style = dict(width = "80%")),
                        html.Label(id = "inertia_weight_text", style = dict(width = "20%", textAlign = "right")),
                        ],
                        style = dict(display = "flex", flexFlow = "row nowrap"),
                        ),
                    dcc.Slider(
                        id = "inertia_weight_slider",
                        value = 0.72,
                        min = 0.,
                        max = 1.,
                        step = 0.01,
                        marks = {0: "", 0.5: "", 1: ""},
                        updatemode = "drag",
                        ),
                            
                    # Cognition parameter
                    html.Div([
                        html.Label("Cognition parameter", style = dict(width = "80%")),
                        html.Label(id = "cognition_parameter_text", style = dict(width = "20%", textAlign = "right")),
                        ],
                        style = dict(display = "flex", flexFlow = "row nowrap"),
                        ),
                    dcc.Slider(
                        id = "cognition_parameter_slider",
                        value = 1.49,
                        min = 0.,
                        max = 3.,
                        step = 0.01,
                        marks = {0: "", 1.5: "", 3: ""},
                        updatemode = "drag",
                        ),
                            
                    # Sociability parameter
                    html.Div([
                        html.Label("Sociability parameter", style = dict(width = "80%")),
                        html.Label(id = "sociability_parameter_text", style = dict(width = "20%", textAlign = "right")),
                        ],
                        style = dict(display = "flex", flexFlow = "row nowrap"),
                        ),
                    dcc.Slider(
                        id = "sociability_parameter_slider",
                        value = 1.49,
                        min = 0.,
                        max = 3.,
                        step = 0.01,
                        marks = {0: "", 1.5: "", 3: ""},
                        updatemode = "drag",
                        ),
                    ],
                    id = "pso_parameters",
                    style = dict(display = "none"),
                    ),
                        
                # CPSO parameters
                html.Div([
                    # Competitivity parameter
                    html.Div([
                        html.Label("Competitivity parameter", style = dict(width = "80%")),
                        html.Label(id = "competitivity_parameter_text", style = dict(width = "20%", textAlign = "right")),
                        ],
                        style = dict(display = "flex", flexFlow = "row nowrap"),
                        ),
                    dcc.Slider(
                        id = "competitivity_parameter_slider",
                        value = 1.25,
                        min = 0.,
                        max = 2.,
                        step = 0.01,
                        marks = {0: "", 1: "", 2: ""},
                        updatemode = "drag",
                        ),
                    ],
                    id = "cpso_parameters",
                    style = dict(display = "none"),
                    ),
                        
                # DE parameters
                html.Div([
                    # Strategy
                    html.Label("Strategy"),
                    dcc.Dropdown(
                        id = "strategy_list",
                        options = destrat_options,
                        value = "best2",
                        multi = False,
                        ),
                    
                    # Crossover probability
                    html.Div([
                        html.Label("Crossover probability", style = dict(width = "80%")),
                        html.Label(id = "crossover_probability_text", style = dict(width = "20%", textAlign = "right")),
                        ],
                        style = dict(display = "flex", flexFlow = "row nowrap", marginTop = "10"),
                        ),
                    dcc.Slider(
                        id = "crossover_probability_slider",
                        value = 0.1,
                        min = 0.,
                        max = 1.,
                        step = 0.01,
                        marks = {0: "", 0.5: "", 1: ""},
                        updatemode = "drag",
                        ),
                            
                    # Differential weight
                    html.Div([
                        html.Label("Differential weight", style = dict(width = "80%")),
                        html.Label(id = "differential_weight_text", style = dict(width = "20%", textAlign = "right")),
                        ],
                        style = dict(display = "flex", flexFlow = "row nowrap"),
                        ),
                    dcc.Slider(
                        id = "differential_weight_slider",
                        value = 0.5,
                        min = 0.,
                        max = 2.,
                        step = 0.01,
                        marks = {0: "", 1: "", 2: ""},
                        updatemode = "drag",
                        ),
                    ],
                    id = "de_parameters",
                    style = dict(display = "none"),
                    ),
                
                # CMAES parameters
                html.Div([
                    # Step size
                    html.Div([
                        html.Label("Step size", style = dict(width = "80%")),
                        html.Label(id = "step_size_text", style = dict(width = "20%", textAlign = "right")),
                        ],
                        style = dict(display = "flex", flexFlow = "row nowrap", marginTop = "10"),
                        ),
                    dcc.Slider(
                        id = "step_size_slider",
                        value = 0,
                        min = -2,
                        max = 2,
                        step = 0.01,
                        marks = { (i): "" for i in (-2, -1, 0, 1, 2) },
                        updatemode = "drag",
                        ),
                    
                    # Step size
                    html.Div([
                        html.Label("Percentage of offsprings", style = dict(width = "80%")),
                        html.Label(id = "mu_perc_text", style = dict(width = "20%", textAlign = "right")),
                        ],
                        style = dict(display = "flex", flexFlow = "row nowrap", marginTop = "10"),
                        ),
                    dcc.Slider(
                        id = "mu_perc_slider",
                        value = 0.5,
                        min = 0.,
                        max = 1.,
                        step = 0.01,
                        marks = {0: "", 0.5: "", 1: ""},
                        updatemode = "drag",
                        ),
                    ],
                    id = "cmaes_parameters",
                    style = dict(display = "none"),
                    ),
                
                # Synchronize checkbox
                html.Br(),
                html.Div([
                    dcc.Checklist(
                        id = "sync_check",
                        options = [ dict(label = "Synchronize", value = "sync") ],
                        values = [ "sync" ],
                        ),
                    ],
                    id = "do_sync",
                    style = dict(display = "none"),
                    ),
                
                # Constrain
                dcc.Checklist(
                    id = "constrain_check",
                    options = [ dict(label = "Constrain", value = "constrain") ],
                    values = [],
                    ),
                ],
                ), 
                            
            html.Div([
                # Optimize button
                html.Button(
                    "Optimize",
                    id = "optimize_button",
                    n_clicks = 0,
                    style = dict(
                        display = "inline-block",
                        marginTop = "15",
                        ),
                    ),
                
                # Play button
                html.Button(
                    "Play",
                    id = "play_button",
                    n_clicks = 0,
                    style = dict(
                        display = "inline-block",
                        marginTop = "15",
                        ),
                    ),
                ],
                style = dict(
                    float = "right",
                    ),
                ),
            
            ],
            style = dict(
                width = "24%",
                display = "table-cell",
                verticalAlign = "middle",
                backgroundColor = "#F5F5F5",
                borderRadius = "6px",
                border = "1px solid #E9E9E9",
                paddingLeft = "15",
                paddingRight = "15",
                paddingTop = "15",
                paddingBottom = "15",
                ),
            ),
        
        # Plot panel
        html.Div([
            dcc.Graph(
                id = "plot_func",
                config = dict(
                    displayModeBar = False,
                    ),
                animate = False,
                ),
            ],
            style = dict(
                width = "74%",
                display = "table-cell",
                verticalAlign = "middle",
                paddingLeft = "25",
                ),
            ),
        ],
        style = dict(
            width = "100%",
            display = "table",
            marginTop = "10",
            marginBottom = "10",
            ),
        ),
                
    # Separator
    html.Hr(
        style = dict(
            marginTop = "0",
            marginBottom = "0",
            marginLeft = "10",
            marginRight = "10",
            ),
        ),
    
    # Footer
    html.Div([
        html.P(
            "This is a preliminary version of the application, it will be updated following Plotly Dash developement.",
            style = dict(
                width = "70%",
                display = "table-cell",
                textAlign = "left",
                ),
        ),
        html.P(
            "Developed by Keurfon Luu",
            style = dict(
                width = "30%",
                display = "table-cell",
                textAlign = "right",
                ),
        ),
        ],
        style = dict(
            width = "100%",
            display = "table",
            marginTop = "10",
            ),
    ),
            
    # Interval component for animation
    dcc.Interval(id = "interval-component", interval = delay),
    
    # Hidden div inside the app that stores intermediates variables
    html.Div(id = "hidden_models", style = dict(display = "none")),
    html.Div(id = "hidden_funcgrid", style = dict(display = "none")),
    html.P("-1", id = "current_iteration", style = dict(display = "none")),
    html.P("0", id = "previous_iteration", style = dict(display = "none")),
    ],
    
    style = dict(
        width = "80%",
        maxWidth = "1200",
        marginLeft = "auto",
        marginRight = "auto",
        backgroundColor = bgcolor,
        padding = "40",
        paddingTop = "20",
        paddingBottom = "20",
        ),
    )
    

@app.callback(
    Output("plot_func", "figure"),
    [ Input("current_iteration", "children") ],
    [ State("hidden_models", "children"),
      State("hidden_funcgrid", "children"),
      State("previous_iteration", "children"),
    ],
    )
def _update_plot(jiter, jmodels, jfuncgrid, piter):
    models_gfit = json.loads(jmodels)
    models = np.array(models_gfit[0])
    gfit = np.array(models_gfit[1])
    funcgrid_xy = json.loads(jfuncgrid)
    funcgrid = np.array(funcgrid_xy[0])
    ax = np.array(funcgrid_xy[1])
    ay = np.array(funcgrid_xy[2])
    max_iter = len(gfit)
    it = (int(jiter) - int(piter)) % max_iter
    trace1 = go.Contour(
        z = funcgrid,
        x = ax,
        y = ay,
        colorscale = "Viridis",
        ncontours = 15,
        showscale = False,
        contours = dict(showlines = True),
        )
    trace2 = go.Scatter(
        x = models[:,0,it],
        y = models[:,1,it],
        mode = "markers",
        marker = dict(
            size = 20,
            color = "White",
            line = dict(width = 1),
            ),
        showlegend = False,
        cliponaxis = True,
        )
    trace3 = go.Scatter(
        x = np.arange(max_iter) + 1,
        y = gfit,
        mode = "lines",
        line = dict(
            width = 1,
            color = "black",
            dash = "dashdot",
            ),
        showlegend = False,
        )
    trace4 = go.Scatter(
        x = np.arange(it) + 1,
        y = gfit[:it],
        mode = "lines",
        line = dict(
            width = 2,
            color = "red",
            ),
        showlegend = False,
        )
    
    figure = make_subplots(rows = 1, cols = 2, print_grid = False)
    figure.append_trace(trace1, 1, 1)
    figure.append_trace(trace2, 1, 1)
    figure.append_trace(trace3, 1, 2)
    figure.append_trace(trace4, 1, 2)
    
    figure["layout"].update(
        title = "Iteration %s" % (it+1),
        font = dict(
            family = "'Open Sans', 'HelveticaNeue', 'Helvetica Neue', Helvetica, Arial, sans-serif",
            size = 16,
            color = "#000000",
            ),
        paper_bgcolor = bgcolor,
        autosize = True,
        margin = dict(l = 50, r = 50, b = 50, t = 50),
        )
    figure["layout"]["xaxis1"].update(
        title = "X1",
        type = "linear",
        range = [ ax[0], ax[-1] ],
        )
    figure["layout"]["yaxis1"].update(
        title = "X2",
        type = "linear",
        range = [ ay[0], ay[-1] ],
        )
    figure["layout"]["xaxis2"].update(
        title = "Iteration number",
        type = "linear",
        range = [ 1, max_iter ],
        )
    figure["layout"]["yaxis2"].update(
        title = "Global best fitness",
        type = "log",
        )
    return figure


@app.callback(
    Output("hidden_models", "children"),
    [ Input("optimize_button", "n_clicks") ],
    [ State("function", "value"),
      State("solver", "value"),
      State("max_iter", "value"),
      State("seed", "value"),
      State("population_size", "value"),
      State("constrain_check", "values"),
      State("inertia_weight_slider", "value"),
      State("cognition_parameter_slider", "value"),
      State("sociability_parameter_slider", "value"),
      State("competitivity_parameter_slider", "value"),
      State("strategy_list", "value"),
      State("crossover_probability_slider", "value"),
      State("differential_weight_slider", "value"),
      State("step_size_slider", "value"),
      State("mu_perc_slider", "value"),
      State("sync_check", "values"),
    ],
    )
def optimize(n_clicks, func, method, max_iter, seed, popsize, constrain,
             w, c1, c2, gamma, strategy, CR, F, sigma, mu_perc, sync):
    bf = BenchmarkFunction(str(func), n_dim = 2)
    solver = Evolutionary(popsize = int(popsize),
                          max_iter = int(max_iter),
                          constrain = "constrain" in constrain,
                          snap = True,
                          random_state = int(seed),
                          **bf.get(),
                          )
    solver.optimize(solver = method,
                    sync = "sync" in sync,
                    w = float(w),
                    c1 = float(c1),
                    c2 = float(c2),
                    gamma = float(gamma),
                    strategy = str(strategy),
                    CR = float(CR),
                    F = float(F),
                    sigma = 10**float(sigma),
                    mu_perc = float(mu_perc),
                    )
    energy = solver.energy
    gfit = [ energy[:,0].min() ]
    for i in range(1, energy.shape[1]):
        gfit.append(min(gfit[i-1], energy[:,i].min()))
    return json.dumps([ solver.models.tolist() ] + [ gfit ])


@app.callback(
    Output("hidden_funcgrid", "children"),
    [ Input("optimize_button", "n_clicks") ],
    [ State("function", "value") ],
    )
def compute_grid(n_clicks, func):
    bf = BenchmarkFunction(str(func), n_dim = 2)
    nx, ny = 101, 101
    mydict = bf.get()
    lower = mydict["lower"]
    upper = mydict["upper"]
    ax = np.linspace(lower[0], upper[0], nx)
    ay = np.linspace(lower[1], upper[1], ny)
    X, Y = np.meshgrid(ax, ay)
    funcgrid = np.array([ mydict["func"]([x, y]) for x, y
                                in zip(X.ravel(), Y.ravel()) ]).reshape((nx, ny))
    return json.dumps([ funcgrid.tolist() ] + [ ax.tolist() ] + [ ay.tolist() ])


@app.callback(
    Output("previous_iteration", "children"),
    [],
    [ State("current_iteration", "children") ],
    events = [ Event("optimize_button", "click") ],
    )
def update_previous_iteration(citer):
    return str(int(citer)+1)


@app.callback(
    Output("current_iteration", "children"),
    [ Input("play_button", "n_clicks") ],
    [ State("current_iteration", "children") ],
    events = [ Event("interval-component", "interval")],
    )
def update_iteration(n_clicks, citer):
    return str(int(citer)+1)


@app.callback(
    Output("play_button", "n_clicks"),
    [ Input("optimize_button", "n_clicks") ],
    [ State("play_button", "n_clicks") ],
    )
def reinit_play(n_clicks_opt, n_clicks_play):
    if n_clicks_play % 2 == 0:
        return 0
    else:
        return 1

    
@app.callback(
    Output("play_button", "children"),
    [],
    [ State("interval-component", "interval") ],
    events = [ Event("play_button", "click") ],
    )
def update_play(interval):
    if interval > 1000:
        return "Pause"
    else:
        return "Play"
    
    
@app.callback(
    Output("interval-component", "interval"),
    [ Input("play_button", "n_clicks") ],
    [ State("interval-component", "interval") ],
    )
def update_interval(n_clicks, interval):
    if n_clicks % 2 == 1:
        return delay
    else:
        return 1e9


@app.callback(
    Output("seed", "value"),
    [ Input("optimize_button", "n_clicks") ],
    [ State("seed_check", "values"), State("seed", "value") ],
    )
def update_seed(n_clicks, fix_seed, current_seed):
    if not "fix_seed" in fix_seed:
        return np.random.randint(MAX_SEED)
    else:
        return current_seed

    
@app.callback(
    Output("cpso_parameters", "style"),
    [ Input("solver", "value") ],
    )
def update_cpso_parameters(solver):
    if str(solver) != "cpso":
        return dict(display = "none")


@app.callback(
    Output("pso_parameters", "style"),
    [ Input("solver", "value") ],
    )
def update_pso_parameters(solver):
    if str(solver) not in [ "pso", "cpso" ]:
        return dict(display = "none")
    
    
@app.callback(
    Output("de_parameters", "style"),
    [ Input("solver", "value") ],
    )
def update_de_parameters(solver):
    if str(solver) != "de":
        return dict(display = "none")
    
    
@app.callback(
    Output("cmaes_parameters", "style"),
    [ Input("solver", "value") ],
    )
def update_cmaes_parameters(solver):
    if str(solver) != "cmaes":
        return dict(display = "none")
    

@app.callback(
    Output("do_sync", "style"),
    [ Input("solver", "value") ],
    )
def update_do_sync(solver):
    if str(solver) not in [ "pso", "cpso", "de" ]:
        return dict(display = "none")
    
# Slider -> inertia_weight_text
@app.callback(
    Output("inertia_weight_text", "children"),
    [ Input("inertia_weight_slider", "value") ],
    )
def update_inertia_weight_text(w):
    return "%.2f" % w


# Slider -> cognition_parameter_text
@app.callback(
    Output("cognition_parameter_text", "children"),
    [ Input("cognition_parameter_slider", "value") ],
    )
def update_cognition_parameter_text(c1):
    return "%.2f" % c1


# Slider -> sociability_parameter_text
@app.callback(
    Output("sociability_parameter_text", "children"),
    [ Input("sociability_parameter_slider", "value") ],
    )
def update_sociability_parameter_text(c2):
    return "%.2f" % c2


# Slider -> competitivity_parameter_text
@app.callback(
    Output("competitivity_parameter_text", "children"),
    [ Input("competitivity_parameter_slider", "value") ],
    )
def update_competitivity_parameter_text(gamma):
    return "%.2f" % gamma


# Slider -> crossover_probability_text
@app.callback(
    Output("crossover_probability_text", "children"),
    [ Input("crossover_probability_slider", "value") ],
    )
def update_crossover_probability_text(CR):
    return "%.2f" % CR


# Slider -> differential_weight_text
@app.callback(
    Output("differential_weight_text", "children"),
    [ Input("differential_weight_slider", "value") ],
    )
def update_differential_weight_text(F):
    return "%.2f" % F


# Slider -> step_size_text
@app.callback(
    Output("step_size_text", "children"),
    [ Input("step_size_slider", "value") ],
    )
def update_step_size_text(sigma):
    return "%.2f" % (10**sigma)


# Slider -> mu_perc_text
@app.callback(
    Output("mu_perc_text", "children"),
    [ Input("mu_perc_slider", "value") ],
    )
def update_mu_perc_text(mu_perc):
    return "%.2f" % mu_perc


#@app.server.route("/static/<path:path>")
#def static_file(path):
#    static_folder = os.path.join(os.getcwd(), 'static')
#    return send_from_directory(static_folder, path)


if __name__ == "__main__":
    app.run_server(debug = True)