import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import pickle
from flask import Flask, request, jsonify
import io
import base64
import os

# Initialize Flask server and Dash app
server = Flask(__name__)
app = dash.Dash(__name__, server=server)

# Generate synthetic sensor data with occasional anomalies
def generate_synthetic_data():
    anomaly_chance = 0.03  # 10% chance to generate an anomaly
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
historical_data = pd.DataFrame(columns=['timestamp', 'sensor_1', 'sensor_2', 'sensor_3', 'anomaly'])

# A DataFrame to store anomalies for display in a table
anomalies_detected = pd.DataFrame(columns=['timestamp', 'sensor_1', 'sensor_2', 'sensor_3', 'anomaly', 'is_anomaly'])

# Flask API to predict anomaly score
@server.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Get data from the POST request
        new_data = np.array([[data['sensor_1'], data['sensor_2'], data['sensor_3']]])  # Format data for prediction

        # Predict anomaly score using the trained model
        anomaly_score = model.decision_function(new_data)
        score = -(anomaly_score[0])
        return jsonify({'anomaly_score': score})  # Return the anomaly score
    except Exception as e:
        return jsonify({'error': str(e)})

# Dash app layout
app.layout = html.Div([
    html.H1("Real-Time Anomaly Detection in Sawmill Trimmer", style={'text-align': 'center', 'font-family': 'Arial, sans-serif'}),
    html.Div([
        html.H4("Current Time in UTC: ", style={'text-align': 'center', 'font-size': '20px', 'font-weight': 'bold'}),
        html.H3(id='current-time-interval', style={'text-align': 'center', 'font-size': '25px', 'font-family': 'Courier New', 'color': 'green'})
    ]),
    html.Div([  # For displaying graphs
        dcc.Graph(id='live-graph-sensors', style={'width': '48%', 'display': 'inline-block'}),
        dcc.Graph(id='live-graph-anomaly', style={'width': '48%', 'display': 'inline-block'})
    ]),
    html.Div([  # For anomaly results and table
        html.H4("Last 10 Detected Anomalies", style={'text-align': 'center', 'color': 'blue'}),
        html.Div(id='anomaly-table-container'),
        html.Button("Download Anomalies CSV", id="download-button", n_clicks=0)
    ]),
    html.Div([  # Hidden div for CSV download
        dcc.Download(id="download-dataframe-csv")
    ]),
    dcc.Interval(
        id='interval-component',
        interval=2 * 1000,  # 2 seconds interval for updating the graphs
        n_intervals=0  # Initial trigger for the interval callback
    )
])

# Update graphs with new sensor data every second
@app.callback(
    [Output('live-graph-sensors', 'figure'),
     Output('live-graph-anomaly', 'figure'),
     Output('current-time-interval', 'children'),
     Output('anomaly-table-container', 'children'),
     Output('download-dataframe-csv', 'data')],
    [Input('interval-component', 'n_intervals'),
     Input('download-button', 'n_clicks')]
)
def update_graphs(n_intervals, download_clicks):
    # Simulate new sensor data
    new_data = generate_synthetic_data()

    # Predict anomaly score using the loaded model
    new_data_array = np.array([[new_data['sensor_1'], new_data['sensor_2'], new_data['sensor_3']]])
    anomaly_score = model.decision_function(new_data_array)[0]

    # Adjust anomaly score and add fluctuation
    adjusted_anomaly_score = -(anomaly_score) + 0.5 + random.uniform(-0.1, 0.1)

    # Add new data to historical data
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    anomaly_label = 'Yes' if adjusted_anomaly_score > 0.7 else 'No'
    historical_data.loc[len(historical_data)] = [timestamp, new_data['sensor_1'], new_data['sensor_2'], new_data['sensor_3'], adjusted_anomaly_score]

    # Add the anomaly to the table of detected anomalies
    anomalies_detected.loc[len(anomalies_detected)] = [timestamp, new_data['sensor_1'], new_data['sensor_2'], new_data['sensor_3'], adjusted_anomaly_score, anomaly_label]

    # Filter data for the last 10 minutes
    time_threshold = datetime.now() - timedelta(minutes=10)
    recent_data = historical_data[historical_data['timestamp'] >= time_threshold.strftime('%Y-%m-%d %H:%M:%S')]

    # Display only anomalies (filter for 'Yes' in the is_anomaly column)
    latest_anomalies = anomalies_detected[anomalies_detected['is_anomaly'] == 'Yes'].tail(10)

    # Update the sensor data graph (for the last 10 minutes)
    sensor_figure = {
        'data': [
            go.Scatter(
                x=recent_data['timestamp'],
                y=recent_data['sensor_1'],
                name='Sensor 1 (Temperature)',
                mode='lines+markers'
            ),
            go.Scatter(
                x=recent_data['timestamp'],
                y=recent_data['sensor_2'],
                name='Sensor 2 (Vibration)',
                mode='lines+markers'
            ),
            go.Scatter(
                x=recent_data['timestamp'],
                y=recent_data['sensor_3'],
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
                x=recent_data['timestamp'],
                y=recent_data['anomaly'],
                name='Anomaly Score',
                mode='lines',
                line={'color': 'green'}
            ),
            go.Scatter(
                x=[timestamp],
                y=[0.7],
                name='Anomaly Threshold',
                mode='lines',
                line={'color': 'red', 'dash': 'dash'}
            )
        ],
        'layout': go.Layout(
            title='Anomaly Score Over Time',
            xaxis=dict(title='Time'),
            yaxis=dict(title='Anomaly Score'),
            showlegend=True
        )
    }

    # Add red star markers for anomalies
    anomaly_points = recent_data[recent_data['anomaly'] > 0.7]
    anomaly_figure['data'].extend([
        go.Scatter(
            x=anomaly_points['timestamp'],
            y=anomaly_points['anomaly'],
            mode='markers+text',
            marker=dict(symbol='star', color='red', size=15),
            text=['⭐'] * len(anomaly_points),
            textposition='bottom center',
            name='Anomaly Detected'
        )
    ])

    # Update current time interval for real-time data
    current_time_interval = f"{timestamp}"

    # Create a table for all detected anomalies (only show anomalies)
    anomaly_table = html.Table([
        html.Tr([html.Th("Timestamp"), html.Th("Sensor 1"), html.Th("Sensor 2"), html.Th("Sensor 3"), html.Th("Anomaly"), html.Th("Is Anomaly")])
    ] + [
        html.Tr([html.Td(row['timestamp']), html.Td(f"{row['sensor_1']:.2f}"), html.Td(f"{row['sensor_2']:.2f}"), html.Td(f"{row['sensor_3']:.2f}"), html.Td(f"{row['anomaly']:.2f}"), html.Td(row['is_anomaly'])])
        for i, row in latest_anomalies.iterrows()
    ], style={
        'width': '100%',
        'border-collapse': 'collapse',
        'margin-top': '20px',
        'font-family': 'Arial, sans-serif',
        'border': '1px solid #ddd',
        'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
        'border-radius': '10px',
        'text-align': 'center'
    })

    # Handle CSV download when button is clicked
    if download_clicks > 0:
        # Convert the anomalies DataFrame to CSV
        csv_string = anomalies_detected.to_csv(index=False, header=True)
        # Encode the CSV string to base64 for downloading
        csv_bytes = io.BytesIO()
        csv_bytes.write(csv_string.encode())
        csv_bytes.seek(0)
        return None, None, None, anomaly_table, dcc.send_data_frame(csv_bytes, "anomalies.csv")

    return sensor_figure, anomaly_figure, current_time_interval, anomaly_table, None

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run_server(debug=True, host='0.0.0.0', port=port,use_reloader=False)
