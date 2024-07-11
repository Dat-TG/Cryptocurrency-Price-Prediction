#                       _oo0oo_
#                      o8888888o
#                      88" . "88
#                      (| -_- |)
#                      0\  =  /0
#                    ___/`---'\___
#                  .' \\|     |// '.
#                 / \\|||  :  |||// \
#                / _||||| -:- |||||- \
#               |   | \\\  -  /// |   |
#               | \_|  ''\---/''  |_/ |
#               \  .-\__  '-'  ___/-. /
#             ___'. .'  /--.--\  `. .'___
#          ."" '<  `.___\_<|>_/___.' >' "".
#         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#         \  \ `_.   \_ __\ /__ _/   .-` /  /
#     =====`-.____`.___ \_____/___.-`___.-'=====
#                       `=---='
#
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#            Phật phù hộ, không bao giờ BUG
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


from datetime import timedelta
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

scaler=MinMaxScaler(feature_range=(0,1))



df_nse = pd.read_csv("./csvdata/BTC-USD.csv")
df_nse["Date"]=pd.to_datetime(df_nse.Date,format="%Y-%m-%d")
df_nse.index=df_nse['Date']

data=df_nse.sort_index(ascending=True,axis=0)
new_dataset=pd.DataFrame(index=range(0,len(df_nse)),columns=['Date','Close'])

for i in range(0,len(data)):
    new_dataset.loc[i, "Date"] = data.iloc[i]['Date']  # Correct assignment using iloc
    new_dataset.loc[i, "Close"] = data.iloc[i]["Close"]

new_dataset.index=new_dataset.Date
new_dataset.drop("Date",axis=1,inplace=True)

final_dataset=new_dataset.values

train_size = int(len(final_dataset) * 0.8)
train_data = final_dataset[:train_size, :]
valid_data = final_dataset[train_size:, :]

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(final_dataset)

x_train_data,y_train_data=[],[]

for i in range(60,len(train_data)):
    x_train_data.append(scaled_data[i-60:i,0])
    y_train_data.append(scaled_data[i,0])

x_train_data,y_train_data=np.array(x_train_data),np.array(y_train_data)

x_train_data=np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))


model = load_model("saved_model.keras", compile=False)
model.compile(optimizer='adam', loss='mean_squared_error')


inputs_data=new_dataset[len(new_dataset)-len(valid_data)-60:].values
inputs_data=inputs_data.reshape(-1,1)
inputs_data=scaler.transform(inputs_data)

X_test=[]
for i in range(60,inputs_data.shape[0]):
    X_test.append(inputs_data[i-60:i,0])
X_test=np.array(X_test)

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_closing_price=model.predict(X_test)
predicted_closing_price=scaler.inverse_transform(predicted_closing_price)

train_data=new_dataset[:train_size]
valid_data=new_dataset[train_size:]
valid_data['Predictions']=predicted_closing_price



df= pd.read_csv("./csvdata/all_data.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df.set_index('Date', inplace=True)

# Create the dropdown dictionary for easy lookup
dropdown_dict = {
    'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum',
    'ADA-USD': 'Cardano',
}

# Create the stock_list from the stock_map
stock_list = [{'label': value, 'value': key} for key, value in dropdown_dict.items()]

# Make future predictions until December 2024
future_dates = pd.date_range(start=df_nse['Date'].max() + timedelta(days=1), end='2024-8-31', freq='B')
future_predictions = []

# Start with the last 60 days of data
last_60_days = scaled_data[-60:].reshape(1, 60, 1)

for date in future_dates:
    predicted_price = model.predict(last_60_days)
    future_predictions.append(predicted_price[0, 0])
    # Add the predicted price to the input data and maintain the shape
    predicted_price_reshaped = predicted_price.reshape(1, 1, 1)
    last_60_days = np.append(last_60_days[:, 1:, :], predicted_price_reshaped, axis=1)

# Inverse transform the predictions to get actual prices
future_predictions = np.array(future_predictions).reshape(-1, 1)
future_predictions = scaler.inverse_transform(future_predictions)

# Create a DataFrame for future predictions
future_df = pd.DataFrame({'Date': future_dates, 'Predictions': future_predictions.flatten()})
future_df.set_index('Date', inplace=True)

def SMA(df, period=50, column="Close"):
    return df[column].rolling(window=period).mean()

def EMA(df, period=50, column="Close"):
    return df[column].ewm(span=period, adjust=False).mean()

def WMA(df, period=50, column="Close"):
    weights = np.arange(1, period + 1)
    return df[column].rolling(window=period).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)

def VWMA(df, period=50, column="Close"):
    volume = df["Volume"] if "Volume" in df.columns else pd.Series(np.ones(len(df)), index=df.index)
    return (df[column] * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()

# Hàm tổng hợp 4 đường MA
def MA(df, period=30, column="Close", ma_type="SMA"):
    if ma_type == "SMA":
        return SMA(df, period, column)
    elif ma_type == "EMA":
        return EMA(df, period, column)
    elif ma_type == "WMA":
        return WMA(df, period, column)
    elif ma_type == "VWMA":
        return VWMA(df, period, column)
    else:
        raise ValueError("Invalid ma_type. Use 'SMA', 'EMA', 'WMA', or 'VWMA'.")

def buy_n_sell(data, col="Close", ctypto_symbol='BTC-USD', period1=20, period2=50, period3=200, MA_type='SMA'):
    df = data[data["symbolid"] == ctypto_symbol]
    name = df["symbolid"].iloc[0]

    df['line1'] = MA(df, period=period1, column="Close", ma_type=MA_type)
    df['line2'] = MA(df, period=period2, column="Close", ma_type=MA_type)
    df['line3'] = MA(df, period=period3, column="Close", ma_type=MA_type)

    # Condition 1
    df['Signal'] = np.where(df["line1"] > df["line2"], 1, 0)
    df['Position'] = df['Signal'].diff()

    df['Buy'] = np.where(df['Position'] == 1, df["Close"], np.nan)
    df['Sell'] = np.where(df['Position'] == -1, df["Close"], np.nan)

    # Condition 2
    df['Golden_Signal'] = np.where(df["line2"] > df["line3"], 1, 0)
    df['Golden_Position'] = df['Golden_Signal'].diff()

    df['Golden_Buy'] = np.where(df['Golden_Position'] == 1, df["Close"], np.nan)
    df['Death_Sell'] = np.where(df['Golden_Position'] == -1, df["Close"], np.nan)

    # Create candlestick chart and buy/sell signals
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df["Close"],
                                 name="Close",
                                 opacity=0.5))

    # Short-term MA
    fig.add_trace(go.Scatter(x=df.index,
                             y=df['line1'],
                             mode='lines',
                             name=f'Short-term MA {period1}',
                             line=dict(color='royalblue')))

    # Medium-term MA
    fig.add_trace(go.Scatter(x=df.index,
                             y=df['line2'],
                             mode='lines',
                             name=f'Medium-term MA {period2}',
                             line=dict(color='darkorange')))

    # Long-term MA
    fig.add_trace(go.Scatter(x=df.index,
                             y=df['line3'],
                             mode='lines',
                             name=f'Long-term MA {period3}',
                             line=dict(color='seagreen')))

    # Buy Signal
    fig.add_trace(go.Scatter(x=df.index[df['Position'] == 1],
                             y=df["Close"][df['Position'] == 1],
                             mode='markers',
                             marker=dict(symbol='triangle-up', color='green', size=12),
                             name='Buy Signal'))

    # Sell Signal
    fig.add_trace(go.Scatter(x=df.index[df['Position'] == -1],
                             y=df["Close"][df['Position'] == -1],
                             mode='markers',
                             marker=dict(symbol='triangle-down', color='red', size=12),
                             name='Sell Signal'))

    # Golden Buy Signal
    fig.add_trace(go.Scatter(x=df.index[df['Golden_Position'] == 1],
                             y=df["Close"][df['Golden_Position'] == 1],
                             mode='markers',
                             marker=dict(symbol='triangle-up', color='gold', size=16),
                             name='Golden Buy Signal',
                             visible='legendonly'))

    # Death Sell Signal
    fig.add_trace(go.Scatter(x=df.index[df['Golden_Position'] == -1],
                             y=df["Close"][df['Golden_Position'] == -1],
                             mode='markers',
                             marker=dict(symbol='triangle-down', color='maroon', size=16),
                             name='Death Sell Signal',
                             visible='legendonly'))

    # Set layout for the chart
    fig.update_layout(title=f'<b style="font-size: 20px;">Candlestick chart with Trading Signals for {name}</b>',
                      xaxis_title='Date',
                      yaxis_title="Close",
                      title_x=0.5,)

    return fig

app.layout = html.Div([
   
    html.H1("Cryptocurrency Price Analysis Dashboard", style={"textAlign": "center"}),

    html.Div([
        html.P([html.B("Developed by: "), "Cong-Dat Le"]),
        html.P([html.B("Student ID: "), "20120454"]),
        html.P([html.B("GitHub: "), html.A("Dat-TG", href="https://github.com/Dat-TG")])
    ], style={"border": "1px solid #000", "padding": "10px", "marginBottom": "50px", "textAlign": "center"}),
   
    dcc.Tabs(id="tabs", children=[

        dcc.Tab(label='Trading Signals', children=[
            html.Div([
                dcc.Dropdown(
                    id='bank-dropdown',
                    options= stock_list,
                    value='BTC-USD',
                    style={"width": "200px", "margin": "16px auto"},
                    clearable=False
                ),
                dcc.Dropdown(
                    id='ma-type-dropdown',
                    options=[{'label': ma, 'value': ma} for ma in ['SMA', 'EMA', 'WMA', 'VWMA']],
                    value='SMA',
                    style={"width": "100px", "margin": "16px auto"},
                    clearable=False
                ),
                dcc.Dropdown(
                    id='period1-dropdown',
                    options=[{'label': str(i), 'value': i} for i in [15, 20, 30]],
                    value=20,
                    style={"width": "100px", "margin": "16px auto"},
                    clearable=False
                ),
                dcc.Dropdown(
                    id='period2-dropdown',
                    options=[{'label': str(i), 'value': i} for i in [50, 80, 100]],
                    value=50,
                    style={"width": "100px", "margin": "16px auto"},
                    clearable=False
                ),
                dcc.Dropdown(
                    id='period3-dropdown',
                    options=[{'label': str(i), 'value': i} for i in [120, 150, 200]],
                    value=200,
                    style={"width": "100px", "margin": "16px auto"},
                    clearable=False
                ),
            ], style={"display": "flex", "flexDirection": "row", "flexWrap": "wrap", "justifyContent": "center"}),
            dcc.Graph(id='trading-signals-chart', style={"height": "100vh"})
        ]),
       
        dcc.Tab(label='Predict Cryptocurrency Price', children=[
            html.Div([
                dcc.Dropdown(id='stock-dropdown',
                             options=stock_list,
                             value='BTC-USD',
                             style={"width": "50%", "margin": "16px auto"}),
                html.H2("Actual vs Predicted Closing Prices", style={"textAlign": "center"}),
                dcc.Graph(
                    id="BTC Data",
                    figure={
                        "data": [
                            go.Scatter(
                                x=valid_data.index,
                                y=valid_data["Close"],
                                mode='lines+markers',
                                name='Actual',
                                line=dict(color='blue')
                            ),
                            go.Scatter(
                                x=valid_data.index,
                                y=valid_data["Predictions"],
                                mode='lines+markers',
                                name='Predicted',
                                line=dict(color='red')
                            ),
                            go.Scatter(
                                x=future_df.index,
                                y=future_df["Predictions"],
                                mode='lines+markers',
                                name='Future Predictions',
                                line=dict(color='green')
                            )
                        ],
                        "layout": go.Layout(
                            title='Actual vs Predicted Closing Prices for BTC-USD',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'},
                            legend=dict(x=0, y=1, traceorder='normal'),
                            hovermode='x'
                        )
                    }
                ),
                html.H2("Actual closing price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="Actual Data AAPL",
                    figure={
                        "data": [
                            go.Scatter(
                                x=valid_data.index,
                                y=valid_data["Close"],
                                mode='markers',
                                name='Actual'
                            )
                        ],
                        "layout": go.Layout(
                            title='Scatter plot of Actual Closing Price for BTC-USD',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }
                ),
                html.H2("LSTM Predicted closing price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data AAPL",
                    figure={
                        "data": [
                            go.Scatter(
                                x=valid_data.index,
                                y=valid_data["Predictions"],
                                mode='markers',
                                name='Predicted'
                            )
                        ],
                        "layout": go.Layout(
                            title='Scatter plot of Predicted Closing Price for BTC-USD',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }
                ),
                html.H2("Future Predictions", style={"textAlign": "center"}),
                dcc.Graph(
                    id="Future Predictions",
                    figure={
                        "data": [
                            go.Scatter(
                                x=future_df.index,
                                y=future_df["Predictions"],
                                mode='markers',
                                name='Future Predictions'
                            )
                        ],
                        "layout": go.Layout(
                            title='Future Predictions of Closing Price for BTC-USD',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }
                )
            ])
        ]),
        dcc.Tab(label='Comparison between cryptocurrencies', children=[
            html.Div([
                html.H1("Stocks High vs Lows", 
                        style={'textAlign': 'center'}),
              
                dcc.Dropdown(id='my-dropdown',
                             options=stock_list, 
                             multi=True,value=[stock_list[0]['value']],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='highlow'),
                html.H1("Stocks Market Volume", style={'textAlign': 'center'}),
         
                dcc.Dropdown(id='my-dropdown2',
                             options=stock_list, 
                             multi=True,value=[stock_list[0]['value']],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='volume')
            ], className="container"),
        ])


    ])
])

@app.callback(
    Output('trading-signals-chart', 'figure'),
    [Input('bank-dropdown', 'value'),
     Input('ma-type-dropdown', 'value'),
     Input('period1-dropdown', 'value'),
     Input('period2-dropdown', 'value'),
     Input('period3-dropdown', 'value')]
)
def update_graph(crypto_symbol, ma_type, period1, period2, period3):
    return buy_n_sell(df, col="Close", ctypto_symbol=crypto_symbol, period1=period1, period2=period2, period3=period3, MA_type=ma_type)



@app.callback(
    [Output('BTC Data', 'figure'),
     Output('Actual Data AAPL', 'figure'),
     Output('Predicted Data AAPL', 'figure'),
     Output('Future Predictions', 'figure')],
    [Input('stock-dropdown', 'value')]
)
def update_graph(selected_stock):
    # Ensure selected_stock is not None
    if selected_stock is None:
        # Return empty figures if selected_stock is None
        return {}, {}, {}

    df_stock = df[df["symbolid"] == selected_stock]

    # Ensure data is sorted by index if necessary
    data = df_stock.sort_index(ascending=True, axis=0)
    data["Date"] = pd.to_datetime(data.index, format="%Y-%m-%d")

    new_dataset = pd.DataFrame(index=range(0, len(data)), columns=['Date', 'Close'])

    for i in range(0, len(data)):
        new_dataset.loc[i, "Date"] = data.iloc[i]["Date"]  # Correct assignment using iloc
        new_dataset.loc[i, "Close"] = data.iloc[i]["Close"]

    new_dataset.set_index('Date', inplace=True)

    final_dataset = new_dataset.values

    train_size = int(len(final_dataset) * 0.8)
    train_data = final_dataset[:train_size, :]
    valid_data = final_dataset[train_size:, :]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(final_dataset)

    x_train_data, y_train_data = [], []

    for i in range(60, len(train_data)):
        x_train_data.append(scaled_data[i - 60:i, 0])
        y_train_data.append(scaled_data[i, 0])

    x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)

    x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

    inputs_data = new_dataset[len(new_dataset) - len(valid_data) - 60:].values
    inputs_data = inputs_data.reshape(-1, 1)
    inputs_data = scaler.transform(inputs_data)

    X_test = []
    for i in range(60, inputs_data.shape[0]):
        X_test.append(inputs_data[i - 60:i, 0])
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_closing_price = model.predict(X_test)
    predicted_closing_price = scaler.inverse_transform(predicted_closing_price)

    train_data = new_dataset[:train_size]
    valid_data = new_dataset[train_size:]
    valid_data['Predictions'] = predicted_closing_price

    # Make future predictions until December 2024
    future_dates = pd.date_range(start=df_stock.index.max() + timedelta(days=1), end='2024-8-31', freq='B')
    future_predictions = []

    # Start with the last 60 days of data
    last_60_days = scaled_data[-60:].reshape(1, 60, 1)

    for date in future_dates:
        predicted_price = model.predict(last_60_days)
        future_predictions.append(predicted_price[0, 0])
        # Add the predicted price to the input data and maintain the shape
        predicted_price_reshaped = predicted_price.reshape(1, 1, 1)
        last_60_days = np.append(last_60_days[:, 1:, :], predicted_price_reshaped, axis=1)

    # Inverse transform the predictions to get actual prices
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)

    # Create a DataFrame for future predictions
    future_df = pd.DataFrame({'Date': future_dates, 'Predictions': future_predictions.flatten()})
    future_df.set_index('Date', inplace=True)


    actual_trace = go.Scatter(
        x=valid_data.index,
        y=valid_data["Close"],
        mode='lines+markers',
        name='Actual',
        line=dict(color='blue')
    )

    predicted_trace = go.Scatter(
        x=valid_data.index,
        y=valid_data["Predictions"],
        mode='lines+markers',
        name='Predicted',
        line=dict(color='red')
    )

    future_trace = go.Scatter(
        x=future_df.index,
        y=future_df["Predictions"],
        mode='lines+markers',
        name='Future Predictions',
        line=dict(color='green')
    )

    actual_scatter_trace = go.Scatter(
        x=valid_data.index,
        y=valid_data["Close"],
        mode='markers',
        name='Actual'
    )

    predicted_scatter_trace = go.Scatter(
        x=valid_data.index,
        y=valid_data["Predictions"],
        mode='markers',
        name='Predicted'
    )

    future_scatter_trace = go.Scatter(
        x=future_df.index,
        y=future_df["Predictions"],
        mode='markers',
        name='Future Predictions'
    )

    figure1 = {
        "data": [actual_trace, predicted_trace, future_trace],
        "layout": go.Layout(
            title=f'Actual vs Predicted Closing Prices for {selected_stock}',
            xaxis={'title': 'Date'},
            yaxis={'title': 'Closing Rate'},
            legend=dict(x=0, y=1, traceorder='normal'),
            hovermode='x'
        )
    }

    figure2 = {
        "data": [actual_scatter_trace],
        "layout": go.Layout(
            title=f'Scatter plot of Actual Closing Price for {selected_stock}',
            xaxis={'title': 'Date'},
            yaxis={'title': 'Closing Rate'}
        )
    }

    figure3 = {
        "data": [predicted_scatter_trace],
        "layout": go.Layout(
            title=f'Scatter plot of Predicted Closing Price for {selected_stock}',
            xaxis={'title': 'Date'},
            yaxis={'title': 'Closing Rate'}
        )
    }

    figure4 = {
        "data": [future_scatter_trace],
        "layout": go.Layout(
            title=f'Future Predictions of Closing Price for {selected_stock}',
            xaxis={'title': 'Date'},
            yaxis={'title': 'Closing Rate'}
        )
    }

    return figure1, figure2, figure3, figure4






@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = dropdown_dict
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
          go.Scatter(x=df[df["symbolid"] == stock].index,
                     y=df[df["symbolid"] == stock]["High"],
                     mode='lines', opacity=0.7, 
                     name=f'High {dropdown[stock]}',textposition='bottom center'))
        trace2.append(
          go.Scatter(x=df[df["symbolid"] == stock].index,
                     y=df[df["symbolid"] == stock]["Low"],
                     mode='lines', opacity=0.6,
                     name=f'Low {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Price (USD)"})}
    return figure


@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    dropdown = dropdown_dict
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
          go.Scatter(x=df[df["symbolid"] == stock].index,
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
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M',
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Transactions Volume"})}
    return figure



if __name__=='__main__':
	app.run_server(debug=True)