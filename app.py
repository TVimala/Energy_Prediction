import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template
import datetime
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model = pickle.load(open('energy_prediction.pkl', 'rb')) 

data_sample = pd.read_csv('operational_Cost.csv', index_col='Datetime', parse_dates=True)  
def process_datetime(dt_str):
    dt = datetime.datetime.strptime(dt_str, "%Y-%m-%dT%H:%M")  
    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour
    day_of_week = dt.weekday()  # Equivalent to .dt.dayofweek
    is_weekend = 1 if day_of_week >= 5 else 0  # 1 for Sat/Sun, 0 for weekdays


    if 'data_sample' not in globals():
        # return [year, month, day, hour, minute, 0, 0] 
        return [year, month, day, hour, day_of_week, is_weekend]  

    # Lagged Consumption (Previous day's energy use)
    previous_day = dt - datetime.timedelta(days=1)
    lagged_consumption = data_sample.loc[previous_day.strftime("%Y-%m-%d"), 'COMED_MW'] if previous_day.strftime("%Y-%m-%d") in data_sample.index else data_sample['COMED_MW'].mean()

    # Is_Peak (1 if above 90th percentile, else 0)
    threshold = data_sample['COMED_MW'].quantile(0.9)
    is_peak = 1 if lagged_consumption > threshold else 0

    # return [year, month, day, hour, minute, lagged_consumption, is_peak]  # Now 6 features
    return [year, month, day, hour, day_of_week, is_weekend]  

@app.route("/")
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure JSON data is received
        data = request.get_json()
        print("Received data:", data)  # Log request data

        if "datetime" not in data:
            return jsonify({"error": "Missing datetime field"}), 400

        # Process input features
        datetime_features = process_datetime(data["datetime"])
        print("Processed Features:", datetime_features)  # Debugging

        input_features = np.array([datetime_features]).reshape(1, -1)  
        print("Final Input to Model:", input_features)  # Debugging

        # Predict using the model
        prediction = model.predict(input_features)[0]

        return jsonify({"prediction": round(prediction, 2)})  

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
