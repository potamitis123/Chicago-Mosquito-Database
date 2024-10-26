# Classifiers
#from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import os


def convert_categorical(train, test, columns):
    lbl = preprocessing.LabelEncoder()
    for col in columns:
        lbl.fit(list(train[col].values) + list(test[col].values))
        train[col] = lbl.transform(train[col].values)
        test[col] = lbl.transform(test[col].values)

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

# Optionally, reset the index after sorting
df = df.reset_index(drop=True)
train = df[(df.year <= 2022)]
test = df[(df.year > 2022)]

convert_categorical(train, test, ['TRAP_TYPE', 'Species', 'Trap','Address','COMMUNITY AREA NAME'])  # Convert categorical data to numbers

sel_columns = ['Block', 'Species', 'TRAP_TYPE', 'Trap', 'Latitude', 'Longitude', 'month', 'week', 'NumMosquitos', 'Address', 'COMMUNITY AREA NAME']


from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier

# It will take some time to run it. If in a hurry go straight to HistGradientBoostingClassifier
# Define parameter grids for each classifier
param_grid_extratrees = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

param_grid_xgb = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0]
}

param_grid_gb = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0]
    }

# Define parameter grid for HistGradientBoostingClassifier
param_grid_hgb = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_samples_leaf': [20, 50, 100],
    'l2_regularization': [0.0, 1.0, 10.0],
    #'max_bins': [255, 512],
    'class_weight': [None, 'balanced']  # Grid search on class_weight to handle imbalanced data
}
    
# Set up cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

X_train, y_train = train[sel_columns], train.WnvPresent 
y_true = test.WnvPresent

# ExtraTreesClassifier Grid Search
#et_clf = ExtraTreesClassifier(random_state=42)
#et_grid_search = GridSearchCV(estimator=et_clf, param_grid=param_grid_extratrees, scoring='roc_auc', cv=cv, n_jobs=1, verbose=2)
#et_grid_search.fit(X_train, y_train)
#print("Best parameters for ExtraTreesClassifier:", et_grid_search.best_params_)
#best_model = et_grid_search.best_estimator_

# Best parameters for ExtraTreesClassifier: {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 1000}
# Evaluate the other metrics on the test set
best_model = ExtraTreesClassifier(max_depth = 10, min_samples_leaf = 1, min_samples_split = 5, n_estimators = 1000, class_weight='balanced', random_state=42)
best_model.fit(X_train, y_train)
y_pred = best_model.predict_proba(test[sel_columns])[:, 1]
# y_pred are the predicted probabilities and y_true are the true labels
auc_score = roc_auc_score(y_true, y_pred)
print(f"AUC Score: {auc_score.round(2)}")

# XGBClassifier Grid Search
#xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
#xgb_grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid_xgb, scoring='roc_auc', cv=cv, n_jobs=1, verbose=2)
#xgb_grid_search.fit(X_train, y_train)
#print("Best parameters for XGBClassifier:", xgb_grid_search.best_params_)
#best_model = xgb_grid_search.best_estimator_

# Evaluate the other metrics on the test set
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
best_model = XGBClassifier(learning_rate = 0.01, max_depth = 5, n_estimators = 1000, subsample = 0.6, scale_pos_weight=scale_pos_weight, eval_metric='logloss', random_state=42)
best_model.fit(X_train, y_train)
y_pred = best_model.predict_proba(test[sel_columns])[:, 1]
auc_score = roc_auc_score(y_true, y_pred)
print(f"AUC Score: {auc_score.round(2)}")

# GradientBoostingClassifier Grid Search
#gb_clf = GradientBoostingClassifier(random_state=42)
#gb_grid_search = GridSearchCV(estimator=gb_clf, param_grid=param_grid_gb, scoring='roc_auc', cv=cv, n_jobs=1, verbose=2)
#gb_grid_search.fit(X_train, y_train)
#print("Best parameters for GradientBoostingClassifier:", gb_grid_search.best_params_)
#best_model = gb_grid_search.best_estimator_

# Evaluate the other metrics on the test set
best_model=GradientBoostingClassifier(learning_rate= 0.01, max_depth= 5, n_estimators= 1000, subsample= 0.8, random_state=42)
sample_weights = np.where(y_train == 1, 1.5, 1.0)
best_model.fit(X_train, y_train, sample_weights)
y_pred = best_model.predict_proba(test[sel_columns])[:, 1]
auc_score = roc_auc_score(y_true, y_pred)
print(f"AUC Score: {auc_score.round(2)}")    

# Create the HistGradientBoostingClassifier
#hgb_clf = HistGradientBoostingClassifier(random_state=42)

# Set up GridSearchCV
#hgb_grid_search = GridSearchCV(estimator=hgb_clf, param_grid=param_grid_hgb, scoring='roc_auc', cv=cv, n_jobs=1, verbose=2)
#hgb_grid_search.fit(X_train, y_train)
# Print the best parameters
#print("Best parameters for HistGradientBoostingClassifier:", hgb_grid_search.best_params_)
#best_model = hgb_grid_search.best_estimator_

best_model = HistGradientBoostingClassifier(l2_regularization=10.0, max_depth= 5, min_samples_leaf= 50, learning_rate= 0.1, class_weight='balanced', random_state=42)
best_model.fit(X_train, y_train)
y_pred = best_model.predict_proba(test[sel_columns])[:, 1]
auc_score = roc_auc_score(y_true, y_pred)
print(f"AUC Score: {auc_score.round(2)}")    

# AUC score
fpr, tpr, thresholds = roc_curve(y_true, y_pred)

# Calculate AUC
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')  # Diagonal line for random guessing

# Optional: Highlight an operational point on the curve
threshold = 0.5  # Example threshold
operational_idx = np.argmin(np.abs(thresholds - threshold))
plt.scatter(fpr[operational_idx], tpr[operational_idx], color='red', label=f'Operational point (Threshold = {threshold:.2f})')

# Add labels and title
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve', fontsize=16)
plt.legend(loc='lower right')

# Show plot
plt.grid(True)
plt.show()
