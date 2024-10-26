# -*- coding: utf-8 -*-
"""
Created on Tue Oct  25 12:36:34 2024
https://data.cityofchicago.org/Health-Human-Services/West-Nile-Virus-WNV-Mosquito-Test-Results/jqe8-8r6s/about_data
@author: Dr.Potamitis Ilyas

"""

import numpy as np
import pandas as pd
import datetime
from scipy.stats import multivariate_normal
from sklearn.metrics import roc_auc_score
import os

def add_days_since_aug1(df):
    df['DaysSinceAug1'] = np.array([
        (datetime.datetime(y, m, d) - datetime.datetime(y, 8, 1)).days
        for y, m, d in zip(df['year'], df['month'], df['day'])
    ], dtype=np.float32)
    return df

def regularize_cov(cov_matrix, epsilon=1e-6):
    cov_matrix += np.eye(cov_matrix.shape[0]) * epsilon
    return cov_matrix


os.chdir('C:/python')
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

df['Date'] = pd.to_datetime(df['Date'])

# Format the 'Date' column to 'DD/MM/YYYY'
df['Date'] = df['Date'].dt.strftime('%d/%m/%Y')

df.dropna(inplace=True)

df['Date'] = pd.to_datetime(df['Date'])  # Convert 'Date' column to datetime if it's not already
df['year'] = df['Date'].dt.year  # Extract the year
df['month'] = df['Date'].dt.month  # Extract the month
df['week'] = df['Date'].dt.isocalendar().week  # Extract the ISO week number
df['day'] = df['Date'].dt.day  # Extract the day  

# Sort the DataFrame by the 'Date' column
df = df.sort_values(by='Date')

# Reset index
df = df.reset_index(drop=True)

# If you want to try it on the Kaggle dataset
#train = df[df['year'].isin([2007, 2009, 2011, 2013])]
#test = df[df['year'].isin([2008, 2010, 2012, 2014])]

train = df[(df.year <= 2022)]
test = df[(df.year > 2022)]

# Preprocess data
train = add_days_since_aug1(train)
test = add_days_since_aug1(test)

# Log-transform NumMosquitos
train['LogNumMosquitos'] = np.log1p(train['NumMosquitos'])
test['LogNumMosquitos'] = np.log1p(test['NumMosquitos'])

# Features to use
features = ['DaysSinceAug1', 'LogNumMosquitos']

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
epsilon = 1e-9
evidence = np.maximum(evidence, epsilon)

posterior_prob_wnv1 = posterior_numerator_wnv1 / evidence

# Add predictions to test set
test['Pred_WnvPresent'] = posterior_prob_wnv1
roc_auc_score(test.WnvPresent, test['Pred_WnvPresent'])

# make pretty plots of the classes WNV positive/negative
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import matplotlib.lines as mlines  # For creating legend handles

# Load your train dataset
# train = pd.read_csv('train.csv')  # Uncomment and modify as needed

def add_days_since_aug1(df):
    df['DaysSinceAug1'] = np.array([
        (datetime.datetime(y, m, d) - datetime.datetime(y, 8, 1)).days
        for y, m, d in zip(df['year'], df['month'], df['day'])
    ], dtype=np.float32)
    return df

def calculate_parameters(df):
    data = df[features].values
    mean = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    # Regularize covariance matrix
    cov += np.eye(cov.shape[0]) * 1e-6
    return mean, cov

# Preprocess data
train = add_days_since_aug1(train)
train['LogNumMosquitos'] = np.log1p(train['NumMosquitos'])
features = ['DaysSinceAug1', 'LogNumMosquitos']

# Separate data by class
train_wnv0 = train[train['WnvPresent'] == 0]
train_wnv1 = train[train['WnvPresent'] == 1]

# Calculate parameters
mean_wnv0, cov_wnv0 = calculate_parameters(train_wnv0)
mean_wnv1, cov_wnv1 = calculate_parameters(train_wnv1)

# Grid for contour plots
x_min = train['DaysSinceAug1'].min() - 5
x_max = train['DaysSinceAug1'].max() + 5
y_min = train['LogNumMosquitos'].min() - 0.5
y_max = train['LogNumMosquitos'].max() + 0.5
x, y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
pos = np.dstack((x, y))

# Evaluate PDFs
rv_wnv0 = multivariate_normal(mean_wnv0, cov_wnv0)
rv_wnv1 = multivariate_normal(mean_wnv1, cov_wnv1)
pdf_wnv0 = rv_wnv0.pdf(pos)
pdf_wnv1 = rv_wnv1.pdf(pos)

# Plotting without scatter plots and adding legends for contours
plt.figure(figsize=(12, 8))

# Contour plot for WnvPresent = 0
contour0 = plt.contour(x, y, pdf_wnv0, levels=5, colors='blue', linestyles='dashed')

# Contour plot for WnvPresent = 1
contour1 = plt.contour(x, y, pdf_wnv1, levels=5, colors='red', linestyles='dashed')

# Create custom legend handles
blue_line = mlines.Line2D([], [], color='blue', linestyle='dashed', label='WnvPresent = 0')
red_line = mlines.Line2D([], [], color='red', linestyle='dashed', label='WnvPresent = 1')

# Labels and legend
plt.xlabel('Days Since August 1')
plt.ylabel('Log of NumMosquitos + 1')
plt.title('Bivariate Gaussian Contours')
plt.legend(handles=[blue_line, red_line])
plt.grid(True)
plt.show()

# Difference in PDFs
pdf_diff = pdf_wnv1 - pdf_wnv0
plt.figure(figsize=(12, 8))
contour = plt.contourf(x, y, pdf_diff, levels=20, cmap='RdBu_r')
cbar = plt.colorbar(contour)
cbar.set_label('PDF Difference (WnvPresent=1 - WnvPresent=0)')

# Since we removed scatter plots, we can add a legend explaining the colormap
# Create custom legend handles for the colormap
from matplotlib.patches import Patch
red_patch = Patch(color='red', label='Higher probability for WnvPresent = 1')
blue_patch = Patch(color='blue', label='Higher probability for WnvPresent = 0')

# Labels and legend
plt.xlabel('Days Since August 1')
plt.ylabel('Log of NumMosquitos + 1')
plt.title('Difference in Bivariate Gaussian PDFs')
plt.legend(handles=[red_patch, blue_patch], loc='upper left')
plt.grid(True)
plt.show()
