# -*- coding: utf-8 -*-
"""
Created on Tue Oct  25 12:36:34 2024
https://data.cityofchicago.org/Health-Human-Services/West-Nile-Virus-WNV-Mosquito-Test-Results/jqe8-8r6s/about_data
@author: Dr.Potamitis Ilyas

"""

import numpy as np
import pandas as pd
import datetime
from scipy.stats import multivariate_normal, hypergeom
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_auc_score

def add_days_since_aug1(df):
    """Adds a column representing the number of days since August 1st of each year."""
    df['DaysSinceAug1'] = np.array([
        (datetime.datetime(y, m, d) - datetime.datetime(y, 8, 1)).days
        for y, m, d in zip(df['year'], df['month'], df['day'])
    ], dtype=np.float32)
    return df

def regularize_cov(cov_matrix, epsilon=1e-6):
    """Adds a small value to the diagonal of the covariance matrix to ensure it's positive definite."""
    cov_matrix += np.eye(cov_matrix.shape[0]) * epsilon
    return cov_matrix

# Load data
file_path = 'Chicago_WNV.xlsx'  # Update the file path as needed
df = pd.read_excel(file_path)

# Rename columns to match the Kaggle dataset format (adjust as necessary)
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
}, inplace=True)

df['Date'] = pd.to_datetime(df['Date'])
df.dropna(inplace=True)

# Extract date components
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day

# Sort the DataFrame by the 'Date' column
df = df.sort_values(by='Date')

# Reset index
df = df.reset_index(drop=True)

# if you want to try the Kaggle data set comment out the following
#train = df[df['year'].isin([2007, 2009, 2011, 2013])]
#test = df[df['year'].isin([2008, 2010, 2012, 2014])]

# if you want to try the Kaggle data set comment the following
# Split into training and test sets
train = df[df['year'] <= 2022]
test = df[df['year'] > 2022]

# Preprocess data
train = add_days_since_aug1(train)
test = add_days_since_aug1(test)

# Log-transform NumMosquitos
train['LogNumMosquitos'] = np.log1p(train['NumMosquitos'])
test['LogNumMosquitos'] = np.log1p(test['NumMosquitos'])

# Features to use
features = ['DaysSinceAug1', 'LogNumMosquitos']

# Calculate the global parameters for the hypergeometric test
N = train.shape[0]  # Total number of observations
K = train['WnvPresent'].sum()  # Total number of successes in the population

print(f"Total observations (N): {N}")
print(f"Total successes (K): {K}")

# Group data by 'Trap' and calculate statistics
trap_stats = train.groupby('Trap').agg(
    positive_cases=('WnvPresent', 'sum'),
    total_cases=('WnvPresent', 'count')
).reset_index()
trap_stats['proportion'] = trap_stats['positive_cases'] / trap_stats['total_cases']

# Perform hypergeometric test for each trap
def perform_hypergeometric_test(row):
    k = int(row['positive_cases'])       # Number of successes in the trap
    n = int(row['total_cases'])          # Number of trials in the trap
    # Compute p-value using the hypergeometric test
    # Calculate the cumulative probability of observing k or more successes
    # under the null hypothesis
    p_value = hypergeom.sf(k - 1, N, K, n)
    return p_value

trap_stats['p_value'] = trap_stats.apply(perform_hypergeometric_test, axis=1)

# Adjust p-values for multiple comparisons using Bonferroni correction
trap_stats['p_value_adj'] = multipletests(trap_stats['p_value'], method='bonferroni')[1]

# Determine significance
trap_stats['significant'] = trap_stats['p_value_adj'] < 0.05

# Calculate weights for each trap
def calculate_weight(row):
    if row['significant']:
        weight = row['proportion'] / (K / N)
    else:
        weight = 1.0
    return weight

trap_stats['weight'] = trap_stats.apply(calculate_weight, axis=1)

# Create a mapping from Trap to weight
trap_weight_mapping = trap_stats.set_index('Trap')['weight'].to_dict()

# Assign weights to each row in the training and test datasets
train['Trap_Weight'] = train['Trap'].map(trap_weight_mapping).fillna(1.0)
test['Trap_Weight'] = test['Trap'].map(trap_weight_mapping).fillna(1.0)

# Proceed with modeling
# Separate data by class
train_wnv0 = train[train['WnvPresent'] == 0][features]
train_wnv1 = train[train['WnvPresent'] == 1][features]

# Compute mean and covariance for WnvPresent = 0
mean_wnv0 = train_wnv0.mean().values
cov_wnv0 = np.cov(train_wnv0.T)
cov_wnv0 = regularize_cov(cov_wnv0)

# Compute mean and covariance for WnvPresent = 1
mean_wnv1 = train_wnv1.mean().values
cov_wnv1 = np.cov(train_wnv1.T)
cov_wnv1 = regularize_cov(cov_wnv1)

# Extract test features
X_test = test[features].values

# Create multivariate normal distributions
mvnorm_wnv0 = multivariate_normal(mean=mean_wnv0, cov=cov_wnv0)
mvnorm_wnv1 = multivariate_normal(mean=mean_wnv1, cov=cov_wnv1)

# Compute likelihoods
likelihood_wnv0 = mvnorm_wnv0.pdf(X_test)
likelihood_wnv1 = mvnorm_wnv1.pdf(X_test)

# Compute prior probabilities
prior_wnv0 = (train['WnvPresent'] == 0).mean()
prior_wnv1 = (train['WnvPresent'] == 1).mean()

# Compute posterior probabilities
posterior_numerator_wnv0 = likelihood_wnv0 * prior_wnv0
posterior_numerator_wnv1 = likelihood_wnv1 * prior_wnv1

evidence = posterior_numerator_wnv0 + posterior_numerator_wnv1
epsilon = 1e-9  # Small constant to prevent division by zero
evidence = np.maximum(evidence, epsilon)

posterior_prob_wnv1 = posterior_numerator_wnv1 / evidence

# Add predictions to test set
test['Pred_WnvPresent'] = posterior_prob_wnv1

# Adjust predictions using Trap weights
test['Adjusted_Pred_WnvPresent'] = test['Pred_WnvPresent'] * test['Trap_Weight']
test['Adjusted_Pred_WnvPresent'] = test['Adjusted_Pred_WnvPresent'].clip(0, .99)

# Evaluate adjusted predictions
auc_score = roc_auc_score(test['WnvPresent'], test['Adjusted_Pred_WnvPresent'])
print(f"ROC AUC Score with Adjusted Probabilities: {auc_score:.4f}")

# Optionally, evaluate unadjusted predictions
auc_score_unadjusted = roc_auc_score(test['WnvPresent'], test['Pred_WnvPresent'])
print(f"ROC AUC Score without Adjustment: {auc_score_unadjusted:.4f}")

# Display the first few rows of the test set with predictions
print(test[['Date', 'Trap', 'WnvPresent', 'Pred_WnvPresent', 'Adjusted_Pred_WnvPresent']].head())
