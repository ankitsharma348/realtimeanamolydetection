import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime
import random
import pickle
from flask import Flask, request, jsonify

# Initialize Flask server and Dash app
server = Flask(__name__)
app = dash.Dash(__name__, server=server)

# Generate synthetic sensor data with occasional anomalies
def generate_synthetic_data():
    anomaly_chance = 0.02  # 10% chance to generate an anomaly
    if random.random() < anomaly_chance:
        sensor_data = {
            'sensor_1': np.random.normal(320, 2),  # Abnormal Temperature (high)
            'sensor_2': np.random.normal(310, 5),   # Abnormal Vibration (high)
            'sensor_3': np.random.normal(450, 10)  # Abnormal Pressure (high)
        }
    else:
        sensor_data = {
            'sensor_1': np.random.normal(50, 5),  # Normal Temperature
            'sensor_2': np.random.normal(30, 2),  # Normal Vibration
            'sensor_3': np.random.normal(100, 10) # Normal Pressure
        }
    return sensor_data

# Load the trained anomaly detection model (using pickle)
def load_model():
    with open('anomaly_detection_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Initialize the model
model = load_model()

# Historical data for visualization (initial empty DataFrame)
historical_data = pd.DataFrame(columns=['timestamp', 'sensor_1', 'sensor_2', 'sensor_3', 'anomaly_score'])

# Flask API to predict anomaly score
@server.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Get data from the POST request
        new_data = np.array([[data['sensor_1'], data['sensor_2'], data['sensor_3']]])  # Format data for prediction

        # Predict anomaly score using the trained model
        anomaly_score = model.decision_function(new_data)
        score = -(anomaly_score[0])
        print(score)
        return jsonify({'anomaly_score': score})  # Return the anomaly score
    except Exception as e:
        return jsonify({'error': str(e)})

# Dash app layout
app.layout = html.Div([
    html.H1("Real-Time Anomaly Detection in Sawmill Trimmer", style={'text-align': 'center', 'font-family': 'Arial, sans-serif'}),
    html.Div([
        html.H4("Current Time: ", style={'text-align': 'center', 'font-size': '20px', 'font-weight': 'bold'}),
        html.H3(id='current-time-interval', style={'text-align': 'center', 'font-size': '25px', 'font-family': 'Courier New', 'color': 'green'})
    ]),
    html.Div([
        dcc.Graph(id='live-graph-sensors', style={'width': '48%', 'display': 'inline-block'}),
        dcc.Graph(id='live-graph-anomaly', style={'width': '48%', 'display': 'inline-block'})
    ]),
    dcc.Interval(
        id='interval-component',
        interval=1 * 1000,  # 1 second interval for updating the graphs
        n_intervals=0  # Initial trigger for the interval callback
    )
])

# Update graphs with new sensor data every second
@app.callback(
    [Output('live-graph-sensors', 'figure'),
     Output('live-graph-anomaly', 'figure'),
     Output('current-time-interval', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_graphs(n_intervals):
    # Simulate new sensor data
    new_data = generate_synthetic_data()

    # Predict anomaly score using the loaded model
    new_data_array = np.array([[new_data['sensor_1'], new_data['sensor_2'], new_data['sensor_3']]])
    anomaly_score = model.decision_function(new_data_array)[0]

    # Add new data to historical data
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    historical_data.loc[len(historical_data)] = [timestamp, new_data['sensor_1'], new_data['sensor_2'], new_data['sensor_3'], (0-anomaly_score)]

    # Update the sensor data graph
    sensor_figure = {
        'data': [
            go.Scatter(
                x=historical_data['timestamp'],
                y=historical_data['sensor_1'],
                name='Sensor 1 (Temperature)',
                mode='lines+markers'
            ),
            go.Scatter(
                x=historical_data['timestamp'],
                y=historical_data['sensor_2'],
                name='Sensor 2 (Vibration)',
                mode='lines+markers'
            ),
            go.Scatter(
                x=historical_data['timestamp'],
                y=historical_data['sensor_3'],
                name='Sensor 3 (Pressure)',
                mode='lines+markers'
            )
        ],
        'layout': go.Layout(
            title='Real-Time Sensor Data (Temperature, Vibration, Pressure)',
            xaxis=dict(title='Time'),
            yaxis=dict(title='Sensor Readings'),
            showlegend=True
        )
    }

    # Update the anomaly score graph
    anomaly_figure = {
        'data': [
            go.Scatter(
                x=historical_data['timestamp'],
                y=historical_data['anomaly_score'],
                name='Anomaly Score',
                mode='lines',
                line={'color': 'red'}
            )
        ],
        'layout': go.Layout(
            title='Anomaly Score Over Time',
            xaxis=dict(title='Time'),
            yaxis=dict(title='Anomaly Score'),
            showlegend=True
        )
    }

    # Update current time interval for real-time data
    current_time_interval = f"{timestamp}"

    return sensor_figure, anomaly_figure, current_time_interval

import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run_server(debug=True, host='0.0.0.0', port=port,use_reloader=False)


