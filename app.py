import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime
import random
import pickle
from flask import Flask, request, jsonify
import io
import base64
import csv

# Initialize Flask server and Dash app
server = Flask(__name__)
app = dash.Dash(__name__, server=server)

# Generate synthetic sensor data with occasional anomalies
def generate_synthetic_data():
    anomaly_chance = 0.10  # 2% chance to generate an anomaly
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

# A DataFrame to store anomalies for display in a table
anomalies_detected = pd.DataFrame(columns=['timestamp', 'sensor_1', 'sensor_2', 'sensor_3', 'anomaly_score'])

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
        html.H4("Current Time: ", style={'text-align': 'center', 'font-size': '20px', 'font-weight': 'bold'}),
        html.H3(id='current-time-interval', style={'text-align': 'center', 'font-size': '25px', 'font-family': 'Courier New', 'color': 'green'})
    ]),
    html.Div([  # For displaying graphs
        dcc.Graph(id='live-graph-sensors', style={'width': '48%', 'display': 'inline-block'}),
        dcc.Graph(id='live-graph-anomaly', style={'width': '48%', 'display': 'inline-block'})
    ]),
    html.Div([  # For displaying anomaly results as a modal
        html.H4("High Anomaly Detected", style={'text-align': 'center', 'color': 'red'}),
        html.Div(id='high-anomaly-modal', style={'display': 'none'}),  # Initially hidden
    ]),
    html.Div([  # Table to display all anomalies detected over time
        html.H4("All Detected Anomalies", style={'text-align': 'center', 'color': 'blue'}),
        html.Div(id='anomaly-table-container'),
        html.Button("Download Anomalies CSV", id="download-button", n_clicks=0)
    ]),
    html.Div([  # Hidden div to handle the CSV download
        dcc.Download(id="download-dataframe-csv")
    ]),
    dcc.Interval(
        id='interval-component',
        interval=5 * 1000,  # 1 second interval for updating the graphs
        n_intervals=0  # Initial trigger for the interval callback
    )
])

# Update graphs with new sensor data every second
@app.callback(
    [Output('live-graph-sensors', 'figure'),
     Output('live-graph-anomaly', 'figure'),
     Output('current-time-interval', 'children'),
     Output('high-anomaly-modal', 'children'),
     Output('high-anomaly-modal', 'style'),
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

    # Increase anomaly score by 0.5 and add a random fluctuation between -0.1 and 0.1
    adjusted_anomaly_score = -(anomaly_score) + 0.5 + random.uniform(-0.1, 0.1)

    # Add new data to historical data
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    historical_data.loc[len(historical_data)] = [timestamp, new_data['sensor_1'], new_data['sensor_2'], new_data['sensor_3'], adjusted_anomaly_score]

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

    # Update the anomaly score graph with color based on anomaly score
    if adjusted_anomaly_score > 0.7:
        anomaly_color = 'red'  # Persistent red color when anomaly score is high
    else:
        anomaly_color = 'green'

    anomaly_figure = {
        'data': [
            go.Scatter(
                x=historical_data['timestamp'],
                y=historical_data['anomaly_score'],
                name='Anomaly Score',
                mode='lines',
                line={'color': anomaly_color}
            )
        ],
        'layout': go.Layout(
            title='Anomaly Score Over Time',
            xaxis=dict(title='Time'),
            yaxis=dict(title='Anomaly Score'),
            showlegend=True
        )
    }

    # Add star marker only to the graph when the anomaly score exceeds 0.7
    if adjusted_anomaly_score > 0.7:
        anomaly_figure['data'].append(
            go.Scatter(
                x=[timestamp],
                y=[adjusted_anomaly_score],
                mode='markers+text',
                marker=dict(symbol='star', color='red', size=15),
                text=['⭐'],
                textposition='bottom center',
                name='Anomaly Detected'
            )
        )

    # Update current time interval for real-time data
    current_time_interval = f"{timestamp}"

    # Check if anomaly score is greater than 0.7 and display high anomaly in a modal
    if adjusted_anomaly_score > 0.7:
        modal_content = html.Div([
            html.H4("High Anomaly Detected", style={'color': 'red'}),
            html.P(f"Timestamp: {timestamp}"),
            html.P(f"Sensor 1: {new_data['sensor_1']:.2f}"),
            html.P(f"Sensor 2: {new_data['sensor_2']:.2f}"),
            html.P(f"Sensor 3: {new_data['sensor_3']:.2f}"),
            html.P(f"Anomaly Score: {adjusted_anomaly_score:.2f}"),
            html.Button("Close", id="close-modal-button", n_clicks=0)
        ])
        modal_style = {'display': 'block'}  # Show the modal

        # Add the anomaly to the table of detected anomalies
        anomalies_detected.loc[len(anomalies_detected)] = [timestamp, new_data['sensor_1'], new_data['sensor_2'], new_data['sensor_3'], adjusted_anomaly_score]

    else:
        modal_content = None
        modal_style = {'display': 'none'}  # Hide the modal

    # Create a table for all detected anomalies with improved styling
    anomaly_table = html.Table([
        html.Tr([html.Th("Timestamp"), html.Th("Sensor 1"), html.Th("Sensor 2"), html.Th("Sensor 3"), html.Th("Anomaly Score")]),
    ] + [
        html.Tr([html.Td(row['timestamp']), html.Td(f"{row['sensor_1']:.2f}"), html.Td(f"{row['sensor_2']:.2f}"), html.Td(f"{row['sensor_3']:.2f}"), html.Td(f"{row['anomaly_score']:.2f}")])
        for i, row in anomalies_detected.iterrows()
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
        return None, None, None, None, None, anomaly_table, dcc.send_data_frame(csv_bytes, "anomalies.csv")

    return sensor_figure, anomaly_figure, current_time_interval, modal_content, modal_style, anomaly_table, None

if __name__ == '__main__':
    app.run_server(debug=True, port=5000, use_reloader=False)
