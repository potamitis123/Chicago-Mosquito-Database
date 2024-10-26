# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 10:56:24 2024
https://data.cityofchicago.org/Health-Human-Services/West-Nile-Virus-WNV-Mosquito-Test-Results/jqe8-8r6s/about_data
@author: Dr.Potamitis Ilyas
"""

import matplotlib.pyplot as plt
import pandas as pd
import os
#import scipy
#from scipy.stats import hypergeom
#import datetime
import numpy as np

os.chdir('C:/python')
os.getcwd()

file_path = 'Chicago_WNV.xlsx'
df = pd.read_excel(file_path)
'''
Index(['SEASON YEAR', 'WEEK', 'TEST ID', 'BLOCK', 'TRAP', 'TRAP_TYPE',
       'TEST DATE', 'NUMBER OF MOSQUITOES', 'RESULT', 'SPECIES',
       'COMMUNITY AREA NUMBER', 'COMMUNITY AREA NAME', 'LATITUDE',
       'LONGITUDE'],
      dtype='object')
'''


# Update TRAP information with the community area details found online so that we dont delete the usefull data
trap_data_update = {
    "T249": {
        "COMMUNITY AREA NUMBER": 7,
        "COMMUNITY AREA NAME": "Lincoln Park",
        "LATITUDE": 41.9227,
        "LONGITUDE": -87.6536
    },
    "T240B": {
        "COMMUNITY AREA NUMBER": 7,
        "COMMUNITY AREA NAME": "Lincoln Park",
        "LATITUDE": 41.9227,
        "LONGITUDE": -87.6536
    },
    "T143": {
        "COMMUNITY AREA NUMBER": 10,
        "COMMUNITY AREA NAME": "Norwood Park",
        "LATITUDE": 41.9959,
        "LONGITUDE": -87.8181
    }
}

# Iterate over the trap data to update the relevant rows
for trap, data in trap_data_update.items():
    mask = df['TRAP'] == trap
    for key, value in data.items():
        df.loc[mask, key] = value



# Dropping all rows with missing values in the 'RESULT' column
# Drop rows with missing values in the specified columns
columns_to_check = ['RESULT', 'COMMUNITY AREA NUMBER', 'COMMUNITY AREA NAME', 'LATITUDE', 'LONGITUDE']
df = df.dropna(subset=columns_to_check)
missing_values_per_column = df.isna().sum()

# Displaying the first few rows of the cleaned dataset
df.info()

df.rename(columns={
    'TEST DATE': 'Date',
    'SPECIES': 'Species',
    'BLOCK': 'Address',
    'COMMUNITY AREA NUMBER': 'Block',
    'TRAP': 'Trap',
    'LATITUDE': 'Latitude',
    'LONGITUDE': 'Longitude',
    'NUMBER OF MOSQUITOES': 'NumMosquitos',
    'RESULT': 'WnvPresent',
    'SEASON YEAR': 'year',
    'WEEK': 'week'
}, inplace=True)


# Convert the 'Date' column to datetime format and handle missing values
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Drop rows where 'Date' could not be converted
df.dropna(subset=['Date'], inplace=True)

# Format the 'Date' column in the desired format '%d/%m/%Y'
df['Date'] = df['Date'].dt.strftime('%d/%m/%Y')

# Set 'Date' as the index
df.set_index('Date', inplace=True)

# Sort the dataframe by the index (date)
df.sort_index(inplace=True)


# Extract month and day from the index (which is the Date column) into separate columns
df['month'] = pd.to_datetime(df.index, format='%d/%m/%Y').month
df['day'] = pd.to_datetime(df.index, format='%d/%m/%Y').day

# round GPS coordinates for heatmap 
df.Latitude = df.Latitude.round(3)
df.Longitude = df.Longitude.round(3)

# Each row is a batch
df.WnvPresent.sum()/len(df)*100 # percentage of positive batches

df.Trap.nunique() # unique traps

df[df.NumMosquitos>50].NumMosquitos # Each batch has maximum 50 insects. Two cases 2014-08-07 77, 2022-08-04 61

# Filter the data to include only rows with the specified TRAP_TYPES
filtered_df = df[df['TRAP_TYPE'].isin(['GRAVID', 'CDC', 'OVI', 'SENTINEL'])]

# Count unique traps for each TRAP_TYPE
unique_traps_per_type = filtered_df.groupby('TRAP_TYPE')['Trap'].nunique()

# Display how many trap types we have
unique_traps_per_type

# Plot 1: Histogram of 'year' where 'WnvPresent' is 1
plt.figure(figsize=(10, 6))
df[df['WnvPresent'] == 1]['year'].hist(bins=len(df['year'].unique()), color='darkblue')
plt.title('Distribution of WNV positive batches by Year', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Counts', fontsize=12)
plt.xticks(sorted(df['year'].unique()))  # Ensure the years are displayed in sorted order on x-axis
plt.grid(True)
plt.show()

# Table 1
# Create a summary table with Species as rows and two columns: NumMosquitos and WnvPresent
species_summary = df.groupby('Species').agg(
    Total_Mosquitos=('NumMosquitos', 'sum'),
    Total_WnvPresent=('WnvPresent', 'sum')
).sort_values(by='Total_Mosquitos', ascending=False).reset_index()

# Print the summary table
print(species_summary)
#species_summary.to_excel('species_summary.xlsx', index=False)


# Plot 2
# Plotting the mosquito species trends over time
# Group the data by 'year' and 'Species' to analyze the mosquito catches trends over time
# Grouping data by year and species and summing mosquito counts
species_over_time = df.groupby(['year', 'Species'])['NumMosquitos'].sum().unstack().fillna(0)

# Plotting the mosquito species trends over time with wider lines and larger markers
plt.figure(figsize=(12, 8))
species_over_time.plot(kind='line', figsize=(12, 8), marker='o', linewidth=2.5, markersize=8)
plt.title('Mosquito Species Trends Over Time')
plt.xlabel('Year')
plt.ylabel('Total Number of Mosquitos')
plt.xticks(ticks=species_over_time.index, labels=species_over_time.index.astype(int))
plt.grid(True)

# Adding the legend inside the figure
plt.legend(title='Species', loc='upper right', fontsize=10, frameon=True)

plt.tight_layout()
plt.show()


# Plot 3: Histogram of 'week' where 'WnvPresent' is 1
# Mapping of weeks to approximate month labels (assuming each month has roughly 4 weeks)
weeks_to_months = {
    1: 'Jan', 5: 'Feb', 9: 'Mar', 13: 'Apr', 17: 'May', 21: 'Jun',
    25: 'Jul', 29: 'Aug', 33: 'Sep', 37: 'Oct', 41: 'Nov', 45: 'Dec'
}

# Plot 2: Histogram of 'week' where 'WnvPresent' is 1
fig, ax1 = plt.subplots(figsize=(10, 6))

# Main plot (weeks on the x-axis)
df[df['WnvPresent'] == 1]['week'].hist(bins=52, color='darkgreen', ax=ax1)
ax1.set_title('Distribution of WNV positive cases by Week', fontsize=16)
ax1.set_xlabel('Week of the Year', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_xticks(range(1, 53, 4))  # Tick every 4 weeks on the x-axis
ax1.grid(True)

# Create a secondary x-axis to display month names
ax2 = ax1.twiny()  # Create a twin Axes that shares the same y-axis
ax2.set_xlim(ax1.get_xlim())  # Match the limits of the x-axis
ax2.set_xticks(list(weeks_to_months.keys()))  # Set the positions for the months
ax2.set_xticklabels(list(weeks_to_months.values()))  # Set the month labels
ax2.set_xlabel('Month', fontsize=12)

plt.show()

# Plot 4
# Calculate the total number of samples for each TRAP_TYPE and Species
unique_traps_per_type = df.groupby('TRAP_TYPE')['Trap'].nunique()

total_mosquitoes = df.groupby(['TRAP_TYPE', 'Species'])['NumMosquitos'].sum().unstack(fill_value=0)

normalized_mosquitoes = total_mosquitoes.div(unique_traps_per_type, axis=0).fillna(0)

# Plot the data as parallel bars
ax = normalized_mosquitoes.plot(kind='bar', figsize=(12, 6), colormap='tab20', width=0.8)

# Set labels and title
plt.xlabel('Trap Type', fontsize=12)
plt.ylabel('Average Number of Mosquitoes per Trap', fontsize=12)
plt.title('Effectiveness of Trap Types in Catching Mosquitoes by Species (Normalized by Number of Traps)', fontsize=16)

# Adjust the legend position
plt.legend(title='Species', loc='upper right', bbox_to_anchor=(0.75, 1))

# Rotate x-axis labels for better readability
plt.xticks(rotation=0, ha='right')

# Add grid lines for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout to prevent clipping of tick-labels
plt.tight_layout()

# Show the plot
plt.show()


# Plot 5 - Adjusted for Fair Comparison
# Calculate the total number of samples for each TRAP_TYPE and Species
unique_traps_per_type = df.groupby('TRAP_TYPE')['Trap'].nunique()

total_mosquitoes = df.groupby(['TRAP_TYPE', 'Species'])['WnvPresent'].sum().unstack(fill_value=0)

normalized_mosquitoes = total_mosquitoes.div(unique_traps_per_type, axis=0).fillna(0)

# Plot the data as parallel bars
ax = normalized_mosquitoes.plot(kind='bar', figsize=(12, 6), colormap='tab20', width=0.8)

# Set labels and title
plt.xlabel('Trap Type', fontsize=12)
plt.ylabel('Proportion of WnvPresent Cases (Normalized)', fontsize=12)
plt.title('Effectiveness of Trap Types in Catching Species When WnvPresent (Normalized by Number of Traps)', fontsize=16)

# Place legend inside the figure
plt.legend(title='Species', loc='upper right', bbox_to_anchor=(.75, 1))

# Rotate x-axis labels for better readability
plt.xticks(rotation=0, ha='right')

# Add grid lines for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout to prevent clipping of tick-labels
plt.tight_layout()

# Show the plot
plt.show()


###########
# Plot 6
# Grouping by 'Trap' and aggregating 'WnvPresent' and 'NumMosquitos' 
trap_grouped = df.groupby('Trap').agg({
    'WnvPresent': 'sum',   # Total count of WnvPresent
    'NumMosquitos': 'sum'  # Total number of mosquitoes
}).reset_index()

# Merging trap_grouped with additional information (Location and Trap type) from the original dataframe
top_traps_info = df[['Trap', 'Address', 'TRAP_TYPE']].drop_duplicates()
trap_grouped = pd.merge(trap_grouped, top_traps_info, on='Trap', how='left')

# Sorting by 'WnvPresent' and selecting the top 10 traps
top_traps = trap_grouped.sort_values(by='WnvPresent', ascending=False).head(10)

# Creating a table with Trap ID, Location, and Trap type
top_traps_table = top_traps[['Trap', 'Address', 'TRAP_TYPE']].reset_index()
print(top_traps_table)

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar width and positions
bar_width = 0.4
index = range(len(top_traps))

# Plotting the 'WnvPresent' counts on the left y-axis
ax1.bar(index, top_traps['WnvPresent'], width=bar_width, color='darkblue', label='WnvPresent Count')
ax1.set_xlabel('Trap #', fontsize=12)
ax1.set_ylabel('WnvPresent Count', fontsize=12, color='darkblue')
ax1.tick_params(axis='y', labelcolor='darkblue')

# Setting x-ticks and labels
ax1.set_xticks([i for i in index])
ax1.set_xticklabels(top_traps['Trap'], rotation=45, ha='right')

# Creating the secondary y-axis for 'NumMosquitos'
ax2 = ax1.twinx()
ax2.bar([i + bar_width for i in index], top_traps['NumMosquitos'], width=bar_width, color='darkgreen', label='NumMosquitos Count')
ax2.set_ylabel('NumMosquitos Count', fontsize=12, color='darkgreen')
ax2.tick_params(axis='y', labelcolor='darkgreen')

# Adding a title
plt.title('Top 10 Traps sorted by WNV positive cases and #Mosquitos', fontsize=16)

# Adjust layout for readability
plt.tight_layout()
plt.show()
top_traps_table.to_excel('effective_traps.xlsx', index=False)


# Plot 7: Create a figure for the histogram
# Group by 'COMMUNITY AREA NAME' to count the number of WNV-positive cases per community area
wnv_by_community = df[df['WnvPresent'] == 1].groupby('COMMUNITY AREA NAME').size().sort_values(ascending=False)

# Plot the data
plt.figure(figsize=(12, 8))
wnv_by_community.plot(kind='bar')

# Set title and labels
plt.title('WNV-Positive Cases by Community Area', fontsize=14)
plt.xlabel('Community Area', fontsize=12)
plt.ylabel('Number of WNV-Positive Cases', fontsize=12)

# Rotate the x-axis labels for readability
plt.xticks(rotation=90)

# Show the plot
plt.tight_layout()
plt.grid()
plt.show()

# Plot 8: Heatmaps
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

from scipy.stats import norm

# Step 1: Create a Heatmap with Folium
m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=12)

# Prepare heat data
try:
    heat_data = [[row['Latitude'], row['Longitude'], row['WnvPresent']] for index, row in df.iterrows()]
    HeatMap(heat_data).add_to(m)
except KeyError as e:
    print(f"KeyError: {e}. Ensure your DataFrame contains the required columns.")

# Save the map to an HTML file
m.save("heatmap.html")

# Step 2: Visualizing Isoperimetric Regions with Matplotlib
points = df[['Longitude', 'Latitude']].values
hull = ConvexHull(points)

plt.figure(figsize=(10, 6))
# Use a more distinctive way to represent 'WnvPresent' by modifying the marker style and scaling the sizes
sizes = np.where(df['WnvPresent'] > 0, df['WnvPresent'] * 50, 10)  # Larger sizes for WnvPresent > 0, default size otherwise
colors = np.where(df['WnvPresent'] > 0, 'red', 'blue')  # Different color for WnvPresent
plt.scatter(df['Longitude'], df['Latitude'], c=colors, s=sizes, alpha=0.7, edgecolor='k')
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'r-', lw=2)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Heatmap with Isoperimetric Regions')
plt.grid()
plt.show()

# Plot 9. Visualize outbreaks
df.index = pd.to_datetime(df.index, dayfirst=True)

# We consider an outbreak as a period with 3 or more consecutive weeks of WnvPresent >= 1
weekly_wnv = df['WnvPresent'].resample('W').sum()

outbreak_threshold = 1  # Minimum number of WnvPresent occurrences for a week to be considered
consecutive_weeks = 3  # Minimum number of consecutive weeks to classify as an outbreak

# Create a mask where WnvPresent >= outbreak_threshold
outbreak_mask = weekly_wnv >= outbreak_threshold

# Find the start and end indices of outbreaks
outbreaks = []
start_idx = None
for i in range(len(outbreak_mask)):
    if outbreak_mask.iloc[i] and start_idx is None:
        start_idx = i
    elif not outbreak_mask.iloc[i] and start_idx is not None:
        if i - start_idx >= consecutive_weeks:
            outbreaks.append((start_idx, i - 1))
        start_idx = None

# If the outbreak goes till the end of the data
if start_idx is not None and len(outbreak_mask) - start_idx >= consecutive_weeks:
    outbreaks.append((start_idx, len(outbreak_mask) - 1))

# Step 4: Plot the data with outbreaks highlighted
plt.figure(figsize=(12, 6))
plt.plot(weekly_wnv.index, weekly_wnv.values, color='blue', label='Weekly WnvPresent Count')

# Highlight outbreaks
for start, end in outbreaks:
    plt.axvspan(weekly_wnv.index[start], weekly_wnv.index[end], color='red', alpha=0.3, label='Outbreak' if start == outbreaks[0][0] else "")

# Set x-axis ticks manually for each year in the data range
years = pd.date_range(start=weekly_wnv.index.min(), end=weekly_wnv.index.max(), freq='YS')
plt.xticks(years, [year.year for year in years], rotation=90)

# Add labels, title, and legend
plt.xlabel('Date', fontsize=12)
plt.ylabel('WnvPresent Count', fontsize=12)
plt.title('WNV Outbreaks Over Time', fontsize=16)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
 
# Plot 10: Histogram of 'NumMosquitos' where 'WnvPresent' is 1
# Use default style (white background)
plt.style.use('default')
plt.figure(figsize=(10, 6))
df[df['WnvPresent'] == 1]['NumMosquitos'].hist(bins=20, color='darkblue')
plt.title('Distribution of Mosquitos in batches (when positive for WNV)', fontsize=16)
plt.xlabel('Number of Mosquitos', fontsize=12)
plt.ylabel('Counts', fontsize=12)
plt.legend(['NumMosquitos'])
plt.grid(True)
plt.show()


# Plot 11
#Calculate number of days since Aug 1 using Pandas' vectorized operations
aug1 = pd.to_datetime(df['year'].astype(str) + '-08-01')
df['DaysSinceAug1'] = (pd.to_datetime(df.index, dayfirst=True) - aug1).dt.days  # Correct the subtraction by converting index to datetime

# Calculate number of days since Aug 1 using Pandas' vectorized operations
# Count occurrences of WnvPresent
days_present_counts = df.groupby('DaysSinceAug1')['WnvPresent'].sum()

# Create a new DataFrame to hold the counts
days_df = pd.DataFrame({'DaysSinceAug1': days_present_counts.index, 'WnvPresent': days_present_counts.values})

# Fit a Gaussian distribution to the WnvPresent data
mean = 15
std_dev = days_df['DaysSinceAug1'].std()

# Create an array of x values for the Gaussian curve
x = np.linspace(days_df['DaysSinceAug1'].min(), days_df['DaysSinceAug1'].max(), 100)

# Calculate the Gaussian PDF values
gaussian_pdf = norm.pdf(x, mean, std_dev) * (days_df['WnvPresent'].max() * 0.05)  # Scale for visibility

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(days_df['DaysSinceAug1'], days_df['WnvPresent'], color='darkblue', edgecolor='black', alpha=0.7, label='Wnv Present Count')
plt.plot(x, 800*gaussian_pdf, color='red', linewidth=2, label='Fitted Gaussian PDF')

# Add labels and title
plt.xlabel('Days Since August 1', fontsize=12)
plt.ylabel('Count of Wnv Present', fontsize=12)
plt.title('Distribution of Wnv Present by Days Since August 1 with Gaussian Fit (2007-2024)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add legend
plt.legend()
plt.show()


# Plot 12 in bivariate.py

# Table 3
# Approximating the number of mosquitos per trap based on multirow count:
# Group by 'Species', 'Trap', 'Address', 'Date'
grouped = df.groupby(['Species', 'Trap', 'Address', 'Date'])

# Get the count of each unique combination of 'Species', 'Trap', 'Address', 'Date'
counts = grouped.size().reset_index(name='count')

# Merge the counts back into the original DataFrame
train_with_counts = pd.merge(df, counts, on=['Species', 'Trap', 'Address', 'Date'])

# Create a list of how many times each combination occurs for each c in range(30)
combination_counts = [len(train_with_counts[train_with_counts['count'] == c]) for c in range(1,30)]

# Calculate the mean 'NumMosquitos' for each unique combination that occurs c times
mean_num_mosquitos = [train_with_counts.loc[train_with_counts['count'] == c, 'NumMosquitos'].mean() for c in range(1,30)]

# Data for the table
data = {
    'Multirow Count': list(range(1,14)),
    '# cases': combination_counts[:13],
    'Mean # Mosquitos': mean_num_mosquitos[:13]
}

# Creating the DataFrame
df_table = pd.DataFrame(data)
df_table['Mean # Mosquitos'] = df_table['Mean # Mosquitos'].round(1)

# Display the table
print(df_table.to_string(index=False))


# Figure 13 is created in the file classification_Chicago_WNV.py