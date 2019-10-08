import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import pandas as pd
import pickle


########### Initiate the app - don't touch this part
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title='titanic_logit'

########### Read in the model and dataset ######
file = open('resources/my-titanic-model.pkl', 'rb')
model=pickle.load(file)
file.close()
train=pd.read_pickle('resources/train.pkl')

########### Set up the layout

app.layout = html.Div(children=[
    html.H1('The odds of Titanic survival'),
    html.Div([
        html.Div([
            html.Div([
                html.H6('Passenger class'),
                dcc.Slider(
                    id='slider1',
                    min=1,
                    max=3,
                    step=0.1,
                    marks={i:str(i) for i in range(1, 4)},
                    value=2
                ),
                html.Br(),
            ], className='six columns'),
            html.Div([
                html.H6('Age group'),
                dcc.Slider(
                    id='slider2',
                    min=1,
                    max=5,
                    step=1,
                    marks={i:str(i) for i in range(1, 6)},
                    value=3,
                ),
                html.Br(),
            ], className='six columns'),
            html.Br(),
        ], className='twelve columns'),
    html.Br(),
    html.A('Code on Github', href='https://github.com/austinlasseter/knn_iris_plotly'),
    ])
])

app.config['suppress_callback_exceptions']=True

######### Define Callbacks


# Message callback
@app.callback(Output('message', 'children'),
              [Input('slider1', 'value'),
               Input('slider2', 'value')])
def radio_results(val0, val1):
    new_observation0=[[val0, val1]]
    prediction=model.predict(new_observation0)
    logodds =prediction[0]
    return f'The predicted survival by travel class {val0} and age-group {val1} is {logodds} logodds. In probability is exp(logodds)/1+exp(logodds)'


############ Deploy
if __name__ == '__main__':
    app.run_server(debug=True)
