import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# Load HTML file
with open("trimet_stopevents_2022-12-07.html", "r", encoding='utf-8') as f:
    soup = BeautifulSoup(f, "lxml")

# Read all tables from HTML
tables = pd.read_html(str(soup))

# Combine all tables into one
df = pd.concat(tables, ignore_index=True)

# Lowercase all column names
df.columns = [col.lower() for col in df.columns]

# Rename needed columns
df.rename(columns={
    'trip_number': 'trip_id',
    'arrive_time': 'arrive_secs',
}, inplace=True)

# Convert arrive_secs to timestamp
base_date = datetime(2022, 12, 7)
df['tstamp'] = df['arrive_secs'].apply(lambda x: base_date + timedelta(seconds=int(x)))

# Extract only the required columns
stops_df = df[['trip_id', 'vehicle_number', 'tstamp', 'location_id', 'ons', 'offs']]

# Save to CSV for checking (optional)
stops_df.to_csv("stops_df.csv", index=False)

# Print some basic summary stats
print("Total stop events:", len(stops_df))
print("Unique vehicles:", stops_df['vehicle_number'].nunique())
print("Unique stop locations:", stops_df['location_id'].nunique())
print("Timestamp range:", stops_df['tstamp'].min(), "to", stops_df['tstamp'].max())
num_ons = len(stops_df[stops_df['ons'] >= 1])
print("Stop events with ons >= 1:", num_ons)
print(f"Percentage with at least one passenger boarding: {100 * num_ons / len(stops_df):.2f}%")

