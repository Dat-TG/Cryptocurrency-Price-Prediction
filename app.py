import dash
from dash import dcc
from dash import html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model # type: ignore
from sklearn.preprocessing import MinMaxScaler
import numpy as np

app = dash.Dash()
app.title = "Cryptocurrency Price Analysis Dashboard - Cong-Dat Le (20120454)"
server = app.server

scaler_btc = MinMaxScaler(feature_range=(0,1))
scaler_eth = MinMaxScaler(feature_range=(0,1))
scaler_ada = MinMaxScaler(feature_range=(0,1))

df_btc = pd.read_csv("./csvdata/BTC-USD.csv")
df_eth = pd.read_csv("./csvdata/ETH-USD.csv")
df_ada = pd.read_csv("./csvdata/ADA-USD.csv")

def preprocess_data(df):
    data = df.sort_index(ascending=True, axis=0)
    new_dataset = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])
    for i in range(0, len(data)):
        new_dataset["Date"][i] = data['Date'][i]
        new_dataset["Close"][i] = data["Close"][i]
    new_dataset.index = new_dataset.Date
    new_dataset.drop("Date", axis=1, inplace=True)
    return new_dataset

btc_dataset = preprocess_data(df_btc)
eth_dataset = preprocess_data(df_eth)
ada_dataset = preprocess_data(df_ada)

def prepare_model_data(new_dataset, scaler):

    final_dataset = new_dataset.values
    train_size = int(len(final_dataset) * 0.8)
    train_data = final_dataset[:train_size, :]
    valid_data = final_dataset[train_size:, :]

    scaled_data = scaler.fit_transform(final_dataset)

    x_train_data, y_train_data = [], []
    for i in range(60, len(train_data)):
        x_train_data.append(scaled_data[i-60:i, 0])
        y_train_data.append(scaled_data[i, 0])

    x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
    x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

    return train_data, valid_data, x_train_data, y_train_data, train_size

btc_train, btc_valid, btc_x_train, btc_y_train, btc_train_size = prepare_model_data(btc_dataset, scaler_btc)
eth_train, eth_valid, eth_x_train, eth_y_train, eth_train_size = prepare_model_data(eth_dataset, scaler_eth)
ada_train, ada_valid, ada_x_train, ada_y_train, ada_train_size = prepare_model_data(ada_dataset, scaler_ada)

model = load_model("saved_model.h5")

def get_predictions(new_dataset, valid_data, train_size, scaler):
    inputs_data = new_dataset[len(new_dataset) - len(valid_data) - 60:].values
    inputs_data = inputs_data.reshape(-1, 1)
    inputs_data = scaler.transform(inputs_data)

    X_test = []
    for i in range(60, inputs_data.shape[0]):
        X_test.append(inputs_data[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_closing_price = model.predict(X_test)
    predicted_closing_price = scaler.inverse_transform(predicted_closing_price)

    train_data_new=new_dataset[:train_size]
    valid_data_new=new_dataset[train_size:]

    valid_data_new['Predictions'] = predicted_closing_price
    return train_data_new, valid_data_new

btc_train, btc_valid = get_predictions(btc_dataset, btc_valid, btc_train_size, scaler_btc)
eth_train, eth_valid = get_predictions(eth_dataset, eth_valid, eth_train_size, scaler_eth)
ada_train, ada_valid = get_predictions(ada_dataset, ada_valid, ada_train_size, scaler_ada)

df = pd.read_csv("./csvdata/all_data.csv")

app.layout = html.Div([
    html.H1("Cryptocurrency Price Analysis Dashboard", style={"textAlign": "center"}),

    html.Div([
        html.P([html.B("Developed by: "), "Cong-Dat Le"]),
        html.P([html.B("Student ID: "), "20120454"]),
        html.P([html.B("GitHub: "), html.A("Dat-TG", href="https://github.com/Dat-TG")])
    ], style={"border": "1px solid #000", "padding": "10px", "marginBottom": "50px", "textAlign": "center"}),

    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Bitcoin', children=[
            html.Div([
                html.H2("Bitcoin Closing Prices", style={"textAlign": "center"}),
                dcc.Graph(
                    id="BTC Data",
                    figure={
                        "data": [
                            go.Scatter(
                                x=btc_valid.index,
                                y=btc_valid["Close"],
                                mode='lines+markers',
                                name='Actual',
                                line=dict(color='blue')
                            ),
                            go.Scatter(
                                x=btc_valid.index,
                                y=btc_valid["Predictions"],
                                mode='lines+markers',
                                name='Predicted',
                                line=dict(color='red')
                            )
                        ],
                        "layout": go.Layout(
                            title='Actual vs Predicted Closing Prices',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'},
                            legend=dict(x=0, y=1, traceorder='normal'),
                            hovermode='x'
                        )
                    }
                ),
                html.H2("Actual closing price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="Actual Data BTC",
                    figure={
                        "data": [
                            go.Scatter(
                                x=btc_valid.index,
                                y=btc_valid["Close"],
                                mode='markers',
                                name='Actual'
                            )
                        ],
                        "layout": go.Layout(
                            title='Scatter plot of Actual Closing Price',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }
                ),
                html.H2("LSTM Predicted closing price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data BTC",
                    figure={
                        "data": [
                            go.Scatter(
                                x=btc_valid.index,
                                y=btc_valid["Predictions"],
                                mode='markers',
                                name='Predicted'
                            )
                        ],
                        "layout": go.Layout(
                            title='Scatter plot of Predicted Closing Price',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }
                )
            ])
        ]),
        dcc.Tab(label='Ethereum', children=[
            html.Div([
                html.H2("Ethereum Closing Prices", style={"textAlign": "center"}),
                dcc.Graph(
                    id="ETH Data",
                    figure={
                        "data": [
                            go.Scatter(
                                x=eth_valid.index,
                                y=eth_valid["Close"],
                                mode='lines+markers',
                                name='Actual',
                                line=dict(color='blue')
                            ),
                            go.Scatter(
                                x=eth_valid.index,
                                y=eth_valid["Predictions"],
                                mode='lines+markers',
                                name='Predicted',
                                line=dict(color='red')
                            )
                        ],
                        "layout": go.Layout(
                            title='Actual vs Predicted Closing Prices',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'},
                            legend=dict(x=0, y=1, traceorder='normal'),
                            hovermode='x'
                        )
                    }
                ),
                html.H2("Actual closing price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="Actual Data ETH",
                    figure={
                        "data": [
                            go.Scatter(
                                x=eth_valid.index,
                                y=eth_valid["Close"],
                                mode='markers',
                                name='Actual'
                            )
                        ],
                        "layout": go.Layout(
                            title='Scatter plot of Actual Closing Price',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }
                ),
                html.H2("LSTM Predicted closing price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data ETH",
                    figure={
                        "data": [
                            go.Scatter(
                                x=eth_valid.index,
                                y=eth_valid["Predictions"],
                                mode='markers',
                                name='Predicted'
                            )
                        ],
                        "layout": go.Layout(
                            title='Scatter plot of Predicted Closing Price',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }
                )
            ])
        ]),
        dcc.Tab(label='Cardano', children=[
            html.Div([
                html.H2("Cardano Closing Prices", style={"textAlign": "center"}),
                dcc.Graph(
                    id="ADA Data",
                    figure={
                        "data": [
                            go.Scatter(
                                x=ada_valid.index,
                                y=ada_valid["Close"],
                                mode='lines+markers',
                                name='Actual',
                                line=dict(color='blue')
                            ),
                            go.Scatter(
                                x=ada_valid.index,
                                y=ada_valid["Predictions"],
                                mode='lines+markers',
                                name='Predicted',
                                line=dict(color='red')
                            )
                        ],
                        "layout": go.Layout(
                            title='Actual vs Predicted Closing Prices',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'},
                            legend=dict(x=0, y=1, traceorder='normal'),
                            hovermode='x'
                        )
                    }
                ),
                html.H2("Actual closing price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="Actual Data ADA",
                    figure={
                        "data": [
                            go.Scatter(
                                x=ada_valid.index,
                                y=ada_valid["Close"],
                                mode='markers',
                                name='Actual'
                            )
                        ],
                        "layout": go.Layout(
                            title='Scatter plot of Actual Closing Price',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }
                ),
                html.H2("LSTM Predicted closing price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data ADA",
                    figure={
                        "data": [
                            go.Scatter(
                                x=ada_valid.index,
                                y=ada_valid["Predictions"],
                                mode='markers',
                                name='Predicted'
                            )
                        ],
                        "layout": go.Layout(
                            title='Scatter plot of Predicted Closing Price',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }
                )
            ])
        ]),
        dcc.Tab(label='Comparison between cryptocurrencies', children=[
            html.Div([
                html.H1("High vs Lows", style={'textAlign': 'center'}),
                dcc.Dropdown(id='my-dropdown',
                             options=[{'label': 'Bitcoin', 'value': 'BTC-USD'},
                                      {'label': 'Ethereum', 'value': 'ETH-USD'},
                                      {'label': 'Cardano', 'value': 'ADA-USD'},],
                             multi=True, value=['BTC-USD'],
                             style={"display": "block", "margin-left": "auto",
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='highlow'),
                html.H1("Stocks Market Volume", style={'textAlign': 'center'}),
                dcc.Dropdown(id='my-dropdown2',
                             options=[{'label': 'Bitcoin', 'value': 'BTC-USD'},
                                      {'label': 'Ethereum', 'value': 'ETH-USD'},
                                      {'label': 'Cardano', 'value': 'ADA-USD'},],
                             multi=True, value=['BTC-USD'],
                             style={"display": "block", "margin-left": "auto",
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='volume')
            ], className="container"),
        ])
    ])
])

@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "ADA-USD": "Cardano"}
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
            go.Scatter(x=df[df["symbolid"] == stock]["Date"],
                       y=df[df["symbolid"] == stock]["High"],
                       mode='lines', opacity=0.7,
                       name=f'High {dropdown[stock]}', textposition='bottom center'))
        trace2.append(
            go.Scatter(x=df[df["symbolid"] == stock]["Date"],
                       y=df[df["symbolid"] == stock]["Low"],
                       mode='lines', opacity=0.6,
                       name=f'Low {dropdown[stock]}', textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1',
                                            '#FF7400', '#FFF400', '#FF0056'],
                                  height=600,
                                  title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
                                  xaxis={"title": "Date",
                                         'rangeselector': {'buttons': list([{'count': 1, 'label': '1M',
                                                                             'step': 'month',
                                                                             'stepmode': 'backward'},
                                                                            {'count': 6, 'label': '6M',
                                                                             'step': 'month',
                                                                             'stepmode': 'backward'},
                                                                            {'step': 'all'}])},
                                         'rangeslider': {'visible': True}, 'type': 'date'},
                                  yaxis={"title": "Price (USD)"})}
    return figure

@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    dropdown = {"BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "ADA-USD": "Cardano"}
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
            go.Scatter(x=df[df["symbolid"] == stock]["Date"],
                       y=df[df["symbolid"] == stock]["Volume"],
                       mode='lines', opacity=0.7,
                       name=f'Volume {dropdown[stock]}', textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1',
                                            '#FF7400', '#FFF400', '#FF0056'],
                                  height=600,
                                  title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
                                  xaxis={"title": "Date",
                                         'rangeselector': {'buttons': list([{'count': 1, 'label': '1M',
                                                                             'step': 'month',
                                                                             'stepmode': 'backward'},
                                                                            {'count': 6, 'label': '6M',
                                                                             'step': 'month',
                                                                             'stepmode': 'backward'},
                                                                            {'step': 'all'}])},
                                         'rangeslider': {'visible': True}, 'type': 'date'},
                                  yaxis={"title": "Transactions Volume"})}
    return figure

if __name__ == '__main__':
    app.run_server(debug=True)
