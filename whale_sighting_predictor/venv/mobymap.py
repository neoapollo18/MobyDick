import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Load the dataset
data = pd.read_csv('/Users/charles/Documents/Python_Files/Moby Dick/whale_sighting_predictor/venv/23305_RWSAS (1).csv')

# Convert SIGHTINGDATE to datetime
data['SIGHTINGDATE'] = pd.to_datetime(data['SIGHTINGDATE'], format='%d-%b-%y')

# Extract the month for grouping
data['month'] = data['SIGHTINGDATE'].dt.month

# Filter the necessary columns
data_filtered = data[['LAT', 'LON', 'month']]

# Calculate the average location for each month
monthly_avg_location = data_filtered.groupby('month').mean().reset_index()

# Initialize the plot
fig, ax = plt.subplots(figsize=(15, 10))

# Set up the Basemap
m = Basemap(projection='mill', llcrnrlat=20, urcrnrlat=60, llcrnrlon=-80, urcrnrlon=-10, resolution='c')

# Draw coastlines and countries
m.drawcoastlines()
m.drawcountries()

# Draw lat/lon grid lines
m.drawparallels(range(20, 61, 10), labels=[1, 0, 0, 0], fontsize=10)
m.drawmeridians(range(-80, -9, 10), labels=[0, 0, 0, 1], fontsize=10)

# Plot the average locations
x, y = m(monthly_avg_location['LON'].values, monthly_avg_location['LAT'].values)
m.scatter(x, y, marker='o', color='red', s=100, zorder=5)

# Add labels
for i, row in monthly_avg_location.iterrows():
    plt.text(x[i], y[i], f"{int(row['month']):02d}", fontsize=12, ha='right', color='blue')

# Title and show the plot
plt.title('Average Whale Sighting Locations by Month')
plt.show()
