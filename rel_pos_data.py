import pandas as pd
from scipy.stats import binomtest, chi2_contingency, ttest_1samp

# --------------------------
# Load stop events data
# --------------------------
stops_df = pd.read_csv("stops_df.csv")

# --------------------------
# Overall Boarding Rate Bias
# --------------------------
total_stops = len(stops_df)
total_boarding_stops = len(stops_df[stops_df['ons'] >= 1])
p_overall = total_boarding_stops / total_stops
print(f"Overall system-wide boarding rate: {p_overall:.4f}")

results_binom = []
for vehicle_id, group in stops_df.groupby('vehicle_number'):
    n = len(group)
    x = len(group[group['ons'] >= 1])
    result = binomtest(x, n, p_overall, alternative='two-sided')
    if result.pvalue < 0.05:
        results_binom.append((vehicle_id, x, n, x/n, result.pvalue))

print("\nVehicles with statistically biased boarding behavior (p < 0.05):")
print("vehicle_id\tx\tn\tboarding_rate\tp_value")
for r in results_binom:
    print(f"{r[0]}\t\t{r[1]}\t{r[2]}\t{r[3]:.2f}\t\t{r[4]:.4f}")

# --------------------------
# Chi-Squared Offs/Ons Bias
# --------------------------
total_ons = stops_df['ons'].sum()
total_offs = stops_df['offs'].sum()
print(f"\nSystem-wide ons: {total_ons}")
print(f"System-wide offs: {total_offs}")

results_chi2 = []
for vehicle_id, group in stops_df.groupby('vehicle_number'):
    vehicle_ons = group['ons'].sum()
    vehicle_offs = group['offs'].sum()
    if vehicle_ons + vehicle_offs < 10:
        continue
    contingency = [
        [vehicle_ons, vehicle_offs],
        [total_ons - vehicle_ons, total_offs - vehicle_offs]
    ]
    chi2, p_value, _, _ = chi2_contingency(contingency)
    if p_value < 0.05:
        results_chi2.append((vehicle_id, vehicle_ons, vehicle_offs, p_value))

print("\nVehicles with biased offs/ons ratios (p < 0.05):")
print("vehicle_id\tons\toffs\tp_value")
for r in results_chi2:
    print(f"{r[0]}\t\t{r[1]}\t{r[2]}\t{r[3]:.4f}")

# --------------------------
# Analysis for Location 6913
# --------------------------
print("\n--- Location 6913 Analysis ---")
location_df = stops_df[stops_df['location_id'] == 6913]
num_stops_location = len(location_df)
unique_buses = location_df['vehicle_number'].nunique()
boarding_events = len(location_df[location_df['ons'] >= 1])
boarding_percentage = (boarding_events / num_stops_location) * 100 if num_stops_location > 0 else 0
print("Number of stops at location 6913:", num_stops_location)
print("Number of unique vehicles at location 6913:", unique_buses)
print(f"Percentage of stops with boarding at location 6913: {boarding_percentage:.2f}%")

# --------------------------
# Analysis for Vehicle 4062
# --------------------------
print("\n--- Vehicle 4062 Analysis ---")
vehicle_df = stops_df[stops_df['vehicle_number'] == 4062]
num_stops_vehicle = len(vehicle_df)
total_boarded = vehicle_df['ons'].sum()
total_deboarded = vehicle_df['offs'].sum()
boarding_stops = len(vehicle_df[vehicle_df['ons'] >= 1])
boarding_vehicle_percentage = (boarding_stops / num_stops_vehicle) * 100 if num_stops_vehicle > 0 else 0
print("Number of stops made by vehicle 4062:", num_stops_vehicle)
print("Total passengers boarded on vehicle 4062:", total_boarded)
print("Total passengers deboarded from vehicle 4062:", total_deboarded)
print(f"Percentage of stops with boarding on vehicle 4062: {boarding_vehicle_percentage:.2f}%")

# --------------------------
# GPS Bias Detection (T-Test)
# --------------------------
print("\n--- GPS Bias Detection ---")
relpos_df = pd.read_csv("trimet_relpos_2022-12-07.csv")
relpos_df.columns = [col.strip().upper() for col in relpos_df.columns]

all_relpos = relpos_df['RELPOS'].values
results_ttest = []
for vehicle_id, group in relpos_df.groupby('VEHICLE_NUMBER'):
    vehicle_relpos = group['RELPOS'].values
    if len(vehicle_relpos) < 5:
        continue
    t_stat, p_value = ttest_1samp(vehicle_relpos, popmean=all_relpos.mean())
    if p_value < 0.005:
        results_ttest.append((vehicle_id, len(vehicle_relpos), p_value))

print("Vehicles with GPS bias (p < 0.005):")
print("vehicle_id\tn_samples\tp_value")
for r in results_ttest:
    print(f"{r[0]}\t\t{r[1]}\t\t{r[2]:.6f}")

