# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 10:56:24 2024
https://data.cityofchicago.org/Health-Human-Services/West-Nile-Virus-WNV-Mosquito-Test-Results/jqe8-8r6s/about_data
@author: Dr.Potamitis Ilyas
This is refactored code (https://github.com/Cardal/Kaggle_WestNileVirus) with leaderboard feedback removed. There may be differences compared to the original code

"""

import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import roc_curve, auc, roc_auc_score
import scipy
from scipy.stats import hypergeom
import datetime
import numpy as np
from sklearn import preprocessing


def convert_species(train, test):
    '''
    Convert the Species field into 4 attributes: IsPipiens, IsPipiensRestuans, IsRestuans, and IsOther.
    (gets 0.5 for Pipiens and for Restuans),
    IsRestuans, and IsOther (for all other species).
    '''
    def process(df):
        df['IsPipiens'] = (df['Species'] == 'CULEX PIPIENS').astype(int) + (df['Species'] == 'CULEX PIPIENS/RESTUANS') * 0.5
        df['IsPipiensRestuans'] = (df['Species'] == 'CULEX PIPIENS/RESTUANS').astype(int)
        df['IsRestuans'] = (df['Species'] == 'CULEX RESTUANS').astype(int) + (df['Species'] == 'CULEX PIPIENS/RESTUANS') * 0.5
        df['IsOther'] = (~df['Species'].isin(['CULEX PIPIENS', 'CULEX PIPIENS/RESTUANS', 'CULEX RESTUANS'])).astype(int)

    # Apply the function to both train and test datasets
    process(train)
    process(test)

def convert_categorical(train, test, columns):
    lbl = preprocessing.LabelEncoder()
    for col in columns:
        lbl.fit(list(train[col].values) + list(test[col].values))
        train[col] = lbl.transform(train[col].values)
        test[col] = lbl.transform(test[col].values)

def add_features(train, test):
    '''
    Add two simple features to the train and test datasets:
    DaysSinceAug1 - number of days since Aug 1
    Norm - normal approximation of the distribution of WnvPresent along a year (based on training data)
    '''
    # Combine both datasets for easy manipulation
    for df in [train, test]:
        # Create a new 'date' column from year, month, and day
        df.loc[:, 'date'] = pd.to_datetime(df[['year', 'month', 'day']])

        # Calculate number of days since Aug 1 using Pandas' vectorized operations
        aug1 = pd.to_datetime(df['year'].astype(str) + '-08-01')
        #df['DaysSinceAug1'] = (df['date'] - aug1).dt.days  # Directly get the number of days
        df.loc[:, 'DaysSinceAug1'] = (df['date'] - aug1).dt.days
        
def calc_mult_counts_prob(test, mult_entries_func):
    '''
    Compute probability factors based on multirow counts.
    test - the test dataset
    mult_entries_func - a function that takes as input the multirow count and returns the factor;
       the predictions are eventually multiplied by these factors, so mult_entries_func should
       typically return 1.0 for count=1, and >1.0 for count>1
    '''
    # Group by 'Species', 'Trap', 'Address', 'Date' to count occurrences
    counts_test = test.groupby(['Species', 'Trap', 'Address', 'Date']).size().reset_index(name='count')
    
    # Merge the counts back into the original test DataFrame
    test_with_counts = pd.merge(test, counts_test, on=['Species', 'Trap', 'Address', 'Date'], how='left')
    
    # Apply the mult_entries_func to the 'count' column to compute the probability factors
    test_with_counts['mult_prob'] = test_with_counts['count'].apply(mult_entries_func)
    
    # Return the computed probabilities as a NumPy array
    return test_with_counts['mult_prob'].values

# Apply per-year and per-species coefficients
def get_year_coeffs(test):
    '''
    Get coefficient for each test year.
    Coefficients were initially set according to multirow counts.
    '''
    # Group by 'Species', 'Trap', 'Address', 'Date' to count occurrences
    counts_test = test.groupby(['Species', 'Trap', 'Address', 'Date']).size().reset_index(name='count')
    
    # Merge the counts back into the original test DataFrame
    test_with_counts = pd.merge(test, counts_test, on=['Species', 'Trap', 'Address', 'Date'], how='left')
    q = [test_with_counts[test_with_counts.year==y]['count'].sum() for y in [2023, 2024]]
    
    return {2023: q[0]/np.mean(q), 2024: q[1]/np.mean(q)}

def get_pipiens_coeffs(test):
    '''
    Get coefficients for species=Pipiens (per year).
    Coefficients were set according to multirow counts.
    '''
    # Group by 'Species', 'Trap', 'Address', 'Date' to count occurrences
    counts_test = test.groupby(['Species', 'Trap', 'Address', 'Date']).size().reset_index(name='count')
    # Merge the counts back into the original test DataFrame
    test_with_counts = pd.merge(test, counts_test, on=['Species', 'Trap', 'Address', 'Date'], how='left')
    q = [test_with_counts[(test_with_counts['year'] == year) & (test_with_counts['Species'] == 'CULEX PIPIENS')]['count'].sum() for year in [2023,2024]]

    return {2023: q[0]/np.mean(q), 2024: q[1]/np.mean(q)}

def get_restuans_coeffs(test):
    '''
    Get coefficients for species=Restuans (per year).
    Coefficients were set according to multirow counts.
    '''
    # Group by 'Species', 'Trap', 'Address', 'Date' to count occurrences
    counts_test = test.groupby(['Species', 'Trap', 'Address', 'Date']).size().reset_index(name='count')
    # Merge the counts back into the original test DataFrame
    test_with_counts = pd.merge(test, counts_test, on=['Species', 'Trap', 'Address', 'Date'], how='left')
    q = [test_with_counts[(test_with_counts['year'] == year) & (test_with_counts['Species'] == 'CULEX RESTUANS')]['count'].sum() for year in [2023,2024]]

    return {2023: q[0]/np.mean(q), 2024: q[1]/np.mean(q)}

def get_pipiens_restuans_coeffs(test):
    '''
    Get coefficients for species=PipiensRestuans (per year).
    Coefficients were set according to multirow counts.
    '''
    # Group by 'Species', 'Trap', 'Address', 'Date' to count occurrences
    counts_test = test.groupby(['Species', 'Trap', 'Address', 'Date']).size().reset_index(name='count')
    # Merge the counts back into the original test DataFrame
    test_with_counts = pd.merge(test, counts_test, on=['Species', 'Trap', 'Address', 'Date'], how='left')
    q = [test_with_counts[(test_with_counts['year'] == year) & (test_with_counts['Species'] == 'CULEX PIPIENS/RESTUANS')]['count'].sum() for year in [2023,2024]]

    return {2023: q[0]/np.mean(q), 2024: q[1]/np.mean(q)}

def get_other_coeffs(df):
    '''
    Get coefficients for species=Other (per year).
    Coefficients are very small since these mosquitoes have very few Wnv cases in the training data.
    '''
    return {2023: 0.01, 2024: 0.01}

def apply_species_coeffs(test, preds, coeffs, species):
    '''
    Update test predictions with species per-year coefficients.
    '''
    return np.array([v * ((1.0 - spc) + spc * coeffs[y]) for v, y, spc in zip(preds, np.array(test['year']), np.array(test[species] == 1))])
    #return np.array([v * ((1.0 - spc) + spc * coeffs[y]) for v, y, spc in zip(preds, np.array(test['year']), np.array(test[species]))])

def apply_yearly_means(test, preds, coeffs):
    '''
    Update test predictions with yearly coefficients.
    '''
    return preds * np.array([coeffs[year] for year in np.asarray(test['year'])])

def count_nearby_mult(ds, dist_func, max_days=7, max_lat=0.1, max_lng=0.1, same_trap=False, same_species=False, use_self=True, ignore_count1=False):
    '''
    Estimate the number of mosquitos per row, based on the multirow count of the row itself and similar rows.
    The number of mosquitos for a specific row with a count of C is taken as num_mosq_per_entry_count[C] (see above).
    These estimates are weighted according to the given dist_func, so the final estimate for the number of mosquitos
    for a specific row is the weighted average of all the nearby rows (where "nearby" is defined by the input 
    parameters and dist_func).
    '''

    # Create a multi-index for counts based on 'Species', 'Trap', 'Address', and 'Date'
    sp_tr_date = ds.set_index(['Species', 'Trap', 'Address', 'Date']).index
    counts = sp_tr_date.value_counts()
    
    # Create a Series for ignoring counts of 1
    cond_ignore1 = counts[sp_tr_date] > 1
    
    # Prepare distances and conditions
    distances = ds[['year', 'DaysSinceAug1', 'Latitude', 'Longitude', 'Species', 'Trap']].copy()
    
    # Create an empty array to store nearby counts
    nearby_mult = np.full(len(ds), num_mosq_per_entry_count[1], dtype=np.float32)  # Default value

    # Create a DataFrame to hold weights and calculations
    for idx in range(len(ds)):
        # Extract current row details
        current_row = distances.iloc[idx]
        y, d, lat, lng, sp, tr = current_row
        
        # Define the conditions for nearby traps
        cond = (distances['year'] == y) & \
               (distances['DaysSinceAug1'].between(d - max_days, d + max_days))
        
        # Check if max_lat and max_lng are not None before using them
        if max_lat is not None:
            cond &= distances['Latitude'].between(lat - max_lat, lat + max_lat)
        if max_lng is not None:
            cond &= distances['Longitude'].between(lng - max_lng, lng + max_lng)
        
        if same_trap:
            cond &= distances['Trap'] == tr
        if same_species:
            cond &= distances['Species'] == sp
        if not use_self:
            cond &= sp_tr_date != sp_tr_date[idx]
        if ignore_count1:
            cond &= cond_ignore1
        
        # Filter the DataFrame based on conditions
        dsc = ds[cond]
        
        if not dsc.empty:
            # Get counts for the nearby rows
            counts_c = counts[sp_tr_date[cond]].values
            
            # If there are valid counts
            num_mosqs = num_mosq_per_entry_count[counts_c]
            weights = dist_func(np.abs(d - dsc['DaysSinceAug1'].values),
                                np.sqrt((lat - dsc['Latitude'].values) ** 2 + (lng - dsc['Longitude'].values) ** 2),
                                np.ones(len(dsc), dtype=np.int32) * sp,
                                dsc['Species'].values)
            
            if np.sum(weights) >= 1e-5:
                weights /= counts_c  # Adjust weights
                nearby_mult[idx] = np.sum(num_mosqs * weights) / np.sum(weights)

    return nearby_mult

def calc_mult_combined_probs(df, mult_counts_probs_test, mult_nearby_probs_test, mult_nearby_probs_test2):
    '''
    Combine the multirow probabilities into a single array of factors with which to multiply the predictions.
    test - the test data
    mult_counts_probs_test - multirow probabilities based on the count of each row
    mult_nearby_probs_test, mult_nearby_probs_test2 - multirow probabilities based on the counts of each
        row and "nearby" rows (i.e., rows from close dates and/or nearby traps)
    '''
    # Group the data by Species, Trap, Address, and Date to calculate the counts
    group_cols = ['Species', 'Trap', 'Address', 'Date']
    counts_test = df.groupby(group_cols).size().to_dict()

    # Parameters
    species_fac_pow = 0.3  
    mult1_pow, mult1_mul = 1.2, 50
    mult2_pow = 0.5  

    # Function to calculate the combined probabilities
    def calculate_prob(row, mn, mn2, mc):
        species = row['Species']
        cnt1 = counts_test.get((1, row['Trap'], row['Address'], row['Date']), 1)
        cnt2 = counts_test.get((2, row['Trap'], row['Address'], row['Date']), 1)
        cnt3 = counts_test.get((3, row['Trap'], row['Address'], row['Date']), 1)

        # Multirow count is 1 -> use nearby multirow counts of nearby rows
        if counts_test[(row['Species'], row['Trap'], row['Address'], row['Date'])] == 1:
            if (species == 1 or species == 3) and cnt2 > 1:
                fac = np.max([2.5, mult_entries_func(cnt2)]) ** species_fac_pow
            elif species == 2 and (cnt1 > 1 or cnt3 > 1):
                fac = np.max([2.5, mult_entries_func(np.max([cnt1, cnt3]))]) ** species_fac_pow
            else:
                fac = 1.0
            return fac * np.min([2.0, 1.0 + ((mn - 1.0) ** mult1_pow) * mult1_mul]) * (mn2 ** mult2_pow)
        else:
            # Multirow count is >1 -> use factor derived from the multirow count (but at least 2.5)
            return np.max([2.5, mc])

    # Apply the probability calculation function to each row
    df.loc[:,'mult_combined_probs'] = df.apply(lambda row: calculate_prob(
        row, 
        mult_nearby_probs_test[row.name], 
        mult_nearby_probs_test2[row.name], 
        mult_counts_probs_test[row.name]), axis=1)

    return df['mult_combined_probs'].values

def add_trap_bias(train, test, bias_factor=50.0):
    '''
    Compute the bias per trap and add it as a feature "TrapBias" to the train and test set.
    The bias is based on the occurrence of WnvPresent=1 in the trap, adjusted using statistical significance.
    '''
    
    # Logit function for transformation
    logit = lambda p: np.log(p) - np.log(1 - p)
    #logit = lambda p: np.log(np.clip(p, 1e-10, 1 - 1e-10)) - np.log(np.clip(1 - p, 1e-10, 1 - 1e-10))
    # Compute total counts and WnvPresent counts in the training data
    num_total = len(train)
    num_wnv = train['WnvPresent'].sum()
    
    # Global ratio of WnvPresent = 1
    ratio_wnv = (num_wnv + 0.05) / (num_total + 1.0)
    
    # Group by 'Trap' and calculate the count and sum of WnvPresent per trap
    trap_stats = train.groupby('Trap')['WnvPresent'].agg(
        trap_count='count',
        trap_wnv='sum'
    ).reset_index()

    # Function to calculate bias for each trap
    def calculate_bias(row):
        num = row['trap_count']
        wnv = row['trap_wnv']
        
        if num == 0:
            return 1.0
        
        ratio = (wnv + 0.05) / (num + 1.0)
        
        # Hyper-geometric test to assess statistical significance
        prob = np.min([
            hypergeom.sf(wnv - 1, num_total, num_wnv, num),
            hypergeom.cdf(wnv, num_total, num_wnv, num)
        ])
        
        # Compute the bias adjusted by the statistical significance
        bias = (ratio / ratio_wnv) ** np.clip((logit(1.0 - prob) / bias_factor), 0.0, 1.0)
        return bias

    # Apply the bias calculation to the trap_stats DataFrame
    trap_stats['bias'] = trap_stats.apply(calculate_bias, axis=1)
    
    # Merge the bias values back into the train and test datasets
    train['TrapBias'] = pd.merge(train, trap_stats[['Trap', 'bias']], on='Trap', how='left')['bias']
    test['TrapBias'] = pd.merge(test, trap_stats[['Trap', 'bias']], on='Trap', how='left')['bias']
    
    # Rename the new feature
    train.rename(columns={'bias': 'TrapBias'}, inplace=True)
    test.rename(columns={'bias': 'TrapBias'}, inplace=True)

def get_outbreaks(train, test):
    '''
    Get coefficients for Wnv outbreaks in each test year. Each outbreak spans two weeks.
    Outbreaks coefficients were set according to multirow counts and Wnv distribution in train data.
    '''
    # Group by 'Species', 'Trap', 'Address', 'Date' to count occurrences
    counts_test = train.groupby(['WnvPresent', 'Trap', 'Address', 'Date']).size().reset_index(name='count')
    # Merge the counts back into the original test DataFrame
    test_with_counts = pd.merge(train, counts_test, on=['WnvPresent', 'Trap', 'Address', 'Date'], how='left')
    
    q = np.asarray([
        test_with_counts[
            (test_with_counts['year'] == year) &
            (test_with_counts['WnvPresent']) &  # Filter for WnvPresent being True
            (test_with_counts['month'] == m) &  # Month = June to Oct
            (test_with_counts['day'].between(1, 16))  # Days between 1st and 16th June
        ]['count'].sum()
        for year in range(2007, 2023) for m in range(6,11) 
    ]).reshape(16,5).sum(axis=0) # train_years x train_months
    
    q = q / q.mean()
    
    # set weights according to multirow counts. The wnv from training adapted to multirow counts of test

    counts_test = test.groupby(['Species', 'Trap', 'Address', 'Date']).size().reset_index(name='count')
    
    # Merge the counts back into the original test DataFrame
    test_with_counts = pd.merge(test, counts_test, on=['Species', 'Trap', 'Address', 'Date'], how='left')
    weights = [test_with_counts[test_with_counts.year==y]['count'].sum() for y in [2023, 2024]]
    weights = weights/np.sum(weights)
    weights = {2023:weights[0],2024:weights[1]}

    # Function to calculate weighted outbreak data
    def create_outbreaks(year, weight, q_data):
        outbreaks = []
        months = [6, 7, 8, 9, 10]  # June to October
        
        for i, count in enumerate(q_data):
            month = months[i]
            
            # Apply the weight to the data and divide across multiple dates in the month
            outbreaks.append((year, month, 1, round(count * weight, 2)))   # 1st of the month
            outbreaks.append((year, month, 16, round(count * weight, 2)))   # 16th of the month
    
        return outbreaks
    
    # Create outbreaks for 2023 and 2024 using respective weights
    outbreaks_23 = create_outbreaks(2023, weights[2023], q)
    outbreaks_24 = create_outbreaks(2024, weights[2024], q)
    
    return outbreaks_23 + outbreaks_24

def prp_outbreaks_daily_factors(outbreaks, years, power=2):
    '''
    Prepare daily multipliers given an array of outbreaks.
    Each outbreak is a factor for a specific date. It influences the days before and after it,
    so that close days effectively receive the same factor; days farther away get a factor
    that is a weighted combination of two outbreaks - the weights are determined by the
    parameter 'power'.
    '''
    # Create a DataFrame to hold the outbreak data
    outbreak_data = []

    # Build the DataFrame from the outbreaks input
    for year, month, day, factor in outbreaks:
        outbreak_data.append((year, month, day, factor))

    outbreak_df = pd.DataFrame(outbreak_data, columns=['Year', 'Month', 'Day', 'Factor'])

    # Convert Year, Month, Day columns to integers
    outbreak_df['Year'] = outbreak_df['Year'].astype(int)
    outbreak_df['Month'] = outbreak_df['Month'].astype(int)
    outbreak_df['Day'] = outbreak_df['Day'].astype(int)

    # Initialize a DataFrame to store daily factors and weights
    date_range = pd.date_range(start='2023-05-01', end='2024-10-31')
    factors_df = pd.DataFrame(index=date_range, columns=['Daily_Factor', 'Daily_Weight'], dtype=np.float64).fillna(0)

    # Calculate daily factors and weights based on the outbreak data
    for index, row in outbreak_df.iterrows():
        od = datetime.datetime(int(row['Year']), int(row['Month']), int(row['Day']))
        factor = row['Factor']

        # Update the direct outbreak day
        factors_df.loc[od, 'Daily_Factor'] += factor
        factors_df.loc[od, 'Daily_Weight'] += 1.0

        # Influence surrounding days
        for days_diff in range(-9, 10):  # Check days from -9 to +9
            if days_diff == 0:
                continue

            affected_date = od + datetime.timedelta(days=days_diff)
            if affected_date in factors_df.index:
                days_diff_abs = abs(days_diff)
                influence = factor / (days_diff_abs ** power)
                factors_df.loc[affected_date, 'Daily_Factor'] += influence
                factors_df.loc[affected_date, 'Daily_Weight'] += 1.0 / (days_diff_abs ** power)


    # Calculate the average factors
    factors_df['Daily_Factor'] = factors_df['Daily_Factor'] / factors_df['Daily_Weight'].replace(0, np.nan)

    # Create a nested dictionary structure for the output
    outbreaks_daily_factors = {}

    # Extract unique years from the outbreak DataFrame
    years = outbreak_df['Year'].unique()  # Extract unique years from outbreak_df

    for year in years:
        outbreaks_daily_factors[year] = {}
        for month in range(5, 11):  # Months 5 (May) to 10 (October)
            outbreaks_daily_factors[year][month] = {}
            for day in range(1, 32):  # Days from 1 to 31
                if (year, month, day) in factors_df.index:
                    factor_value = factors_df.loc[datetime.datetime(year, month, day), 'Daily_Factor']
                    outbreaks_daily_factors[year][month][day] = factor_value if not np.isnan(factor_value) else 0.0

    return outbreaks_daily_factors


def apply_outbreaks(test, preds, outbreaks):
    '''
    Update test predictions using outbreaks coefficients.
    '''
    test_years = np.unique(test['year'])
    outbreaks_daily_factors = prp_outbreaks_daily_factors(outbreaks, test_years, power=10)
    
    # Create a DataFrame for the test data
    #test_df = pd.DataFrame(test)
    test_df = test.copy()

    # Initialize a column for daily factors
    test_df['Daily_Factor'] = 1.0  # Default factor is 1.0

    # Update the Daily_Factor using outbreaks_daily_factors
    test_df['Daily_Factor'] = test_df.apply(
        lambda row: outbreaks_daily_factors[row['year']][row['month']][row['day']]
        if row['month'] in outbreaks_daily_factors[row['year']] and row['day'] in outbreaks_daily_factors[row['year']][row['month']]
        else 1.0,  # Fallback if the month or day is not found
        axis=1
    )

    # Update predictions
    return preds * test_df['Daily_Factor'].astype(np.float32)

def get_geo_outbreaks(test):
    '''
    Get coefficients for Wnv occurrence on a geographical basis (per test year).
    '''
    geo_outbreaks = {2023: (1.4, 1, 0.8), 2024: (1.2, 1, 0.7)}
    return geo_outbreaks

def apply_geo_outbreaks(test, preds, geo_outbreaks):
    '''
    Update test predictions using geo-outbreaks coefficients.
    '''
    # Calculate the Latitude - Longitude difference
    test_geo_dists = test['Latitude'] - test['Longitude']

    # Calculate the percentiles for the differences
    lml_perc = np.percentile(test_geo_dists, [83, 50, 17])

    # Calculate absolute distances from percentiles
    test_geo_dists = np.abs(test_geo_dists.values[:, np.newaxis] - lml_perc)  # Shape (n_samples, 3)

    dist_pow = 4

    # Iterate over geo_outbreaks to apply the year-specific bias
    for geo_year, geo_biases in geo_outbreaks.items():

        # Create a weights DataFrame by applying the distance function
        weights_df = pd.DataFrame({
            f'weight_{tg}': 1.0 / ((1e-3) + test_geo_dists[:, tg]) ** dist_pow
            for tg in range(3)
        })

        # Check if the year matches geo_year, and apply the geo_biases if it does
        year_mask = (test['year'] == geo_year)

        # Multiply predictions by weighted average of geo_biases where year matches
        if year_mask.any():
            # Apply the weights to geo_biases
            weighted_geo_biases = weights_df.apply(
                lambda row: np.average(geo_biases, weights=row.values), axis=1
            )
            weighted_geo_biases = weighted_geo_biases.astype(np.float32)
            # Update the preds where the year matches geo_year
            preds[year_mask] *= weighted_geo_biases[year_mask].values

    return preds.astype(np.float32)


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



#df.columns of Kaggle dataset

'''
Index(['Date', 'Address', 'Species', 'Block', 'Street', 'Trap',
       'AddressNumberAndftreet', 'Latitude', 'Longitude', 'AddressAccuracy',
       'NumMosquitos', 'WnvPresent'],
      dtype='object')
'''

# Renaming certain columns
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
    # Add more as needed
}, inplace=True)

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Drop rows where 'Date' could not be converted
df.dropna(subset=['Date'], inplace=True)

# Format the 'Date' column to 'DD/MM/YYYY'
df['Date'] = df['Date'].dt.strftime('%d/%m/%Y')

df.dropna(inplace=True)

df['Date'] = pd.to_datetime(df['Date'])  # Convert 'Date' column to datetime if it's not already
df['year'] = df['Date'].dt.year  # Extract the year
df['month'] = df['Date'].dt.month  # Extract the month
df['week'] = df['Date'].dt.isocalendar().week  # Extract the ISO week number
df['day'] = df['Date'].dt.day  # Extract the day

# Convert the 'Date' column to datetime if it's not already
df['Date'] = pd.to_datetime(df['Date'])

# Sort the DataFrame by the 'Date' column
df = df.sort_values(by='Date')

# Optionally, reset the index after sorting
df = df.reset_index(drop=True)

aug1 = pd.to_datetime(df['year'].astype(str) + '-08-01')
df['DaysSinceAug1'] = (df['Date'] - aug1).dt.days  # Directly get the number of days
# Use Normal distribution as baseline
print('Using Normal distribution as base predictor')
#train[train['WnvPresent'] == 1].DaysSinceAug1.mean() #13.28
#train[train['WnvPresent'] == 1].DaysSinceAug1.std() #18.23
norm_mean, norm_std = 9.5, 18.0 # 9.5
df.loc[:,'Norm'] = scipy.stats.norm.pdf((df['DaysSinceAug1'] - norm_mean) / norm_std)

train = df[(df.year <= 2022)]
test = df[(df.year > 2022)]

print('-> Processing features')
convert_species(train, test)  # Convert species into binary features
convert_categorical(train, test, ['TRAP_TYPE', 'Trap','Address','COMMUNITY AREA NAME'])  # Convert categorical data to numbers

# Add various features
print('-> Adding features')
add_features(train, test)

df.Latitude = df.Latitude.round(3)
df.Longitude = df.Longitude.round(3)

# Prepare predictions for test dataset    
print('-> Preparing test predictions')
test_pred = np.array(test['Norm'])   
roc_auc_score(test.WnvPresent, test_pred)

# print('Applying species yearly biases')
test_year_means = get_year_coeffs(test)
test_pred = apply_yearly_means(test, test_pred, test_year_means)
roc_auc_score(test.WnvPresent, test_pred)

# # It does not offer something for 2023 2024 test set
# pip_year_fac = get_pipiens_coeffs(test)
res_year_fac = get_restuans_coeffs(test)
#prs_year_fac = get_pipiens_restuans_coeffs(test)
otr_year_fac = get_other_coeffs(test)

convert_categorical(train, test, ['Species'])

# test_pred = apply_species_coeffs(test, test_pred, pip_year_fac, 'IsPipiens')
test_pred = apply_species_coeffs(test, test_pred, res_year_fac, 'IsRestuans')
#test_pred = apply_species_coeffs(test, test_pred, prs_year_fac, 'IsPipiensRestuans')
test_pred = apply_species_coeffs(test, test_pred, otr_year_fac, 'IsOther')
roc_auc_score(test.WnvPresent, test_pred)

test.reset_index(inplace=True)

# Not proper refactoring
#outbreaks = get_outbreaks(train, test)
#test_pred = apply_outbreaks(test, test_pred, outbreaks)
#roc_auc_score(test.WnvPresent, test_pred)

geo_outbreaks = get_geo_outbreaks(test)
test_pred = apply_geo_outbreaks(test, test_pred, geo_outbreaks)
roc_auc_score(test.WnvPresent, test_pred) # 

# Compute traps bias from train data, apply it to predictions
print('Computing trap bias')

add_trap_bias(train, test, bias_factor=45.0) #50
test['TrapBias'] = test['TrapBias'].bfill()

trap_bias_pow = 0.2 #0.5
print('Applying trap bias')
test_pred = test_pred * (test['TrapBias'] ** trap_bias_pow)
roc_auc_score(test.WnvPresent, test_pred)

# Compute probabilities according to multirow counts
# In the 2023-2024 multirow counts are different and so is the fit
print('Calculating multirow counts probabilities')
mult_entries_coeff = 3.3 
mult_entries_func = lambda c: 1.0 + mult_entries_coeff * np.log(c)

# Calculate multirow counts probabilities
mult_counts_probs_train = calc_mult_counts_prob(train, mult_entries_func)
mult_counts_probs_test = calc_mult_counts_prob(test, mult_entries_func)

#num_mosq_per_entry_count = 7.0 * (1.0 + 3.2 * np.log(1.0 + np.log(np.arange(100))))
num_mosq_per_entry_count = 8.0 * (1.0 + 5.2 * np.log(1.0 + np.log(np.arange(100))))
# Count nearby multirow counts (=the counts of similar rows) 
print('Counting nearby multirow counts #1 (35 days, dist factor 100000)')
mult_dist_func = lambda dist_days, dist_geo, species1, species2: (1.0 / (1.0 + dist_days)) * (1.0 / (1.0 + 100000 * dist_geo))

# Train. Is needed only if it becomes a training feature
#nearby_mult_train = count_nearby_mult(train, mult_dist_func, max_days=35, max_lat=None, max_lng=None, same_trap=False, same_species=True)
#mult_nearby_probs_train = np.array([mult_entries_func(1.0 + m / 50.0) / mult_entries_func(1.0 + num_mosq_per_entry_count[1] / 50.0) for m in nearby_mult_train])
#assert np.min(mult_nearby_probs_train) >= 1.0 and np.max(mult_nearby_probs_train) < mult_entries_func(2.0)

# Test
nearby_mult_test = count_nearby_mult(test, mult_dist_func, max_days=35, max_lat=None, max_lng=None, same_trap=False, same_species=True)
mult_nearby_probs_test = np.array([mult_entries_func(1.0 + m / 50.0) / mult_entries_func(1.0 + num_mosq_per_entry_count[1] / 50.0) for m in nearby_mult_test])
#assert np.min(mult_nearby_probs_test) >= 1.0 and np.max(mult_nearby_probs_test) < mult_entries_func(2.0)

print('Counting nearby multirow counts #2 (14 days, same trap)')
mult_dist_func2 = lambda dist_days, dist_geo, species1, species2: (1.0 / (1.0 + dist_days) ** 1.5)
# Train
#nearby_mult_train2 = count_nearby_mult(train, mult_dist_func2, max_days=14, max_lat=None, max_lng=None, same_trap=True, same_species=True)
#mult_nearby_probs_train2 = np.array([mult_entries_func(1.0 + m / 50.0) / mult_entries_func(1.0 + num_mosq_per_entry_count[1] / 50.0) for m in nearby_mult_train2])
#assert np.min(mult_nearby_probs_train2) >= 1.0 and np.max(mult_nearby_probs_train2) < mult_entries_func(2.0)
# Test
nearby_mult_test2 = count_nearby_mult(test, mult_dist_func2, max_days=14, max_lat=None, max_lng=None, same_trap=True, same_species=True)
mult_nearby_probs_test2 = np.array([mult_entries_func(1.0 + m / 50.0) / mult_entries_func(1.0 + num_mosq_per_entry_count[1] / 50.0) for m in nearby_mult_test2])
#assert np.min(mult_nearby_probs_test2) >= 1.0 and np.max(mult_nearby_probs_test2) < mult_entries_func(2.0)

# Combine multirow counts probabilities. In the train and test columns mult_combined_probs_test is appended
print('Combining multirow counts probabilities')
# Train
#mult_combined_probs_train = calc_mult_combined_probs(train, mult_counts_probs_train, mult_nearby_probs_train, mult_nearby_probs_train2)
# Test
mult_combined_probs_test = calc_mult_combined_probs(test, mult_counts_probs_test, mult_nearby_probs_test, mult_nearby_probs_test2)

# Apply combined multirow counts probabilities to predictions
print('Applying combined multirow counts probabilities')
test_pred = test_pred * test.mult_combined_probs
roc_auc_score(test.WnvPresent, test_pred)


# Calculate ROC curve and AUC for the first classifier
fpr1, tpr1, thresholds1 = roc_curve(test.WnvPresent, test_pred)
roc_auc1 = auc(fpr1, tpr1)

# Calculate ROC curve and AUC for the second classifier
# You need to run the Tree-based code to get test_pred2
fpr2, tpr2, thresholds2 = roc_curve(test.WnvPresent, test_pred2)
roc_auc2 = auc(fpr2, tpr2)

# Plot ROC curves for both classifiers
plt.figure(figsize=(8, 6))
plt.plot(fpr1, tpr1, color='blue', lw=2, label=f'Probabilistic approach ROC curve (area = {roc_auc1:.2f})')
plt.plot(fpr2, tpr2, color='green', lw=2, label=f'Tree-based classifier ROC curve (area = {roc_auc2:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')  # Diagonal line for random guessing

# Optional: Highlight an operational point on the first classifier's curve
threshold = 0.275  # Example threshold
operational_idx1 = np.argmin(np.abs(thresholds1 - threshold))
plt.scatter(fpr1[operational_idx1], tpr1[operational_idx1], color='red', s=100, label=f'Operational point Classifier 1 (Threshold = {threshold:.2f})')

# Add labels, title, and legend
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve Comparison', fontsize=16)
plt.legend(loc='lower right')

# Show plot
plt.grid(True)
plt.show()

# Plot 12
# Group the train dataset by 'Trap' and calculate the mean TrapBias for each trap
# Compute Mean and Standard Deviation
trap_bias_summary = train.groupby('Trap')['TrapBias'].agg(['mean', 'std']).reset_index()
trap_bias_summary = trap_bias_summary.sort_values(by='mean', ascending=False)

# Select Top N Traps
N = 20
top_traps = trap_bias_summary.head(N)

# Plot the Bar Chart with Larger Bars and Error Bars
plt.figure(figsize=(14, 8))
positions = range(len(top_traps))
bar_width = 0.8

plt.bar(positions, top_traps['mean'], yerr=top_traps['std'], capsize=5,
        color='skyblue', edgecolor='black', alpha=0.8, width=bar_width)

plt.xticks(positions, top_traps['Trap'], rotation=45, ha='right', fontsize=12)
plt.xlabel('Trap', fontsize=14)
plt.ylabel('Mean TrapBias', fontsize=14)
plt.title('Top {} Traps Sorted by Mean TrapBias with Standard Deviation'.format(N), fontsize=16)

# Annotate bars with mean values
for i, value in enumerate(top_traps['mean']):
    plt.text(i, value + 0.02, f"{value:.2f}", ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.show()

