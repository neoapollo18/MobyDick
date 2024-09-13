import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('23305_RWSAS (1).csv')

# Convert SIGHTINGDATE to datetime
data['SIGHTINGDATE'] = pd.to_datetime(data['SIGHTINGDATE'], format='%d-%b-%y')

# Extract additional date features
data['month'] = data['SIGHTINGDATE'].dt.month
data['day'] = data['SIGHTINGDATE'].dt.day
data['day_of_week'] = data['SIGHTINGDATE'].dt.dayofweek

# Drop the CERTAINTY, DUPLICATE, and GROUPSIZE columns
data = data.drop(columns=['Id', 'CERTAINTY', 'DUPLICATE', 'GROUPSIZE'])

# Encode categorical variables
data = pd.get_dummies(data, columns=['CATEGORY', 'MOM_CALF'], drop_first=True)

# Remove rows with missing values
data_cleaned = data.dropna()

# Define features and target variables, excluding the SIGHTINGDATE column
features = data_cleaned.drop(columns=['LAT', 'LON', 'SIGHTINGDATE'])
target_lat = data_cleaned['LAT']
target_lon = data_cleaned['LON']

# Ensure all feature columns are numerical
features = features.apply(pd.to_numeric)

# Split the data into training and testing sets
X_train, X_test, y_lat_train, y_lat_test = train_test_split(features, target_lat, test_size=0.2, random_state=42)
X_train, X_test, y_lon_train, y_lon_test = train_test_split(features, target_lon, test_size=0.2, random_state=42)

# Initialize the Random Forest models
model_lat = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_lon = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# Train the models
model_lat.fit(X_train, y_lat_train)
model_lon.fit(X_train, y_lon_train)

# Predict on the test set
y_lat_pred = model_lat.predict(X_test)
y_lon_pred = model_lon.predict(X_test)

# Evaluate the models
mse_lat = mean_squared_error(y_lat_test, y_lat_pred)
mse_lon = mean_squared_error(y_lon_test, y_lon_pred)

print(f'Mean Squared Error for Latitude: {mse_lat:.2f}')
print(f'Mean Squared Error for Longitude: {mse_lon:.2f}')

# Get user input for new data
user_date = input("Enter the sighting date (YYYY-MM-DD): ")

# Convert user input date to datetime
user_date = pd.to_datetime(user_date)

# Create a DataFrame for the new input data
new_data = pd.DataFrame({
    'SIGHTINGDATE': [user_date]
})

# Extract date features from user input
new_data['month'] = new_data['SIGHTINGDATE'].dt.month
new_data['day'] = new_data['SIGHTINGDATE'].dt.day
new_data['day_of_week'] = new_data['SIGHTINGDATE'].dt.dayofweek

# Include the necessary dummy columns with 0 values for the prediction
dummy_columns = [col for col in data_cleaned.columns if col.startswith('CATEGORY_') or col.startswith('MOM_CALF_')]
for col in dummy_columns:
    new_data[col] = 0

# Drop the SIGHTINGDATE column
new_data = new_data.drop(columns=['SIGHTINGDATE'])

# Ensure all feature columns are numerical
new_data = new_data.apply(pd.to_numeric)

# Predict the latitude and longitude
predicted_lat = model_lat.predict(new_data)
predicted_lon = model_lon.predict(new_data)

print(f'Predicted Latitude: {predicted_lat[0]:.4f}')
print(f'Predicted Longitude: {predicted_lon[0]:.4f}')
