import os
import pandas as pd
from flask import Flask, request, render_template, abort
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

# Load and preprocess the dataset
def load_data():
    try:
        data = pd.read_csv('/Users/charles/Documents/Python_Files/Moby Dick/whale_sighting_predictor/venv/23305_RWSAS (1).csv')
    except FileNotFoundError:
        print("File not found. Please check the dataset path.")
        abort(500, description="File not found. Please check the dataset path.")

    data['SIGHTINGDATE'] = pd.to_datetime(data['SIGHTINGDATE'], format='%d-%b-%y')
    data['month'] = data['SIGHTINGDATE'].dt.month
    data['day'] = data['SIGHTINGDATE'].dt.day
    data['day_of_week'] = data['SIGHTINGDATE'].dt.dayofweek
    data = data.drop(columns=['Id', 'CERTAINTY', 'DUPLICATE', 'GROUPSIZE'])
    data = pd.get_dummies(data, columns=['CATEGORY', 'MOM_CALF'], drop_first=True)
    data_cleaned = data.dropna()
    features = data_cleaned.drop(columns=['LAT', 'LON', 'SIGHTINGDATE'])
    target_lat = data_cleaned['LAT']
    target_lon = data_cleaned['LON']
    features = features.apply(pd.to_numeric)
    return features, target_lat, target_lon

features, target_lat, target_lon = load_data()

# Train the models
X_train_lat, X_test_lat, y_train_lat, y_test_lat = train_test_split(features, target_lat, test_size=0.2, random_state=42)
X_train_lon, X_test_lon, y_train_lon, y_test_lon = train_test_split(features, target_lon, test_size=0.2, random_state=42)

model_lat = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_lon = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

model_lat.fit(X_train_lat, y_train_lat)
model_lon.fit(X_train_lon, y_train_lon)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_date = request.form['sighting_date']
        user_date = pd.to_datetime(user_date)
    except Exception as e:
        return f"Error: {str(e)}", 400

    new_data = pd.DataFrame({
        'SIGHTINGDATE': [user_date],
        'month': [user_date.month],
        'day': [user_date.day],
        'day_of_week': [user_date.dayofweek]
    })

    dummy_columns = [col for col in features.columns if col.startswith('CATEGORY_') or col.startswith('MOM_CALF_')]
    for col in dummy_columns:
        new_data[col] = 0

    new_data = new_data.drop(columns=['SIGHTINGDATE'])
    new_data = new_data.apply(pd.to_numeric)
    predicted_lat = model_lat.predict(new_data)
    predicted_lon = model_lon.predict(new_data)

    return render_template('result.html', latitude=predicted_lat[0], longitude=predicted_lon[0])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
