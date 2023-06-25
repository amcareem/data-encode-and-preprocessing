import pandas as pd
from pandas import read_csv
from sklearn.impute import SimpleImputer

# Declaring header names
winsconsin_headers = ['sample_code', 'c thickness', 'uni_cell_size', 'uni_cell_shape', 'marg_adhesion', 'epi_cell_size', 'nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses', 'tumor_class']

# Read the data using the read_csv class of pandas
wins_data = read_csv("venv/winsconsin_b_cancer.csv", names=winsconsin_headers)

# Set display options to show all columns
pd.set_option('display.max_columns', None)

print(wins_data.shape)  # Print the new data dimension
print("\nFirst 5 Data:")
print(wins_data.head(5))  # Print the first 5 rows
print("\nLast 5 data:")
print(wins_data.tail(5))  # Print the last 5 rows

# Replace '?' with NaN
wins_data['nuclei'].replace('?', float('nan'), inplace=True)

# Convert all variables to float
wins_data = wins_data.astype(float)

# Checking for missing data
win_data_numeric = wins_data.copy()  # Create a copy of the data
win_data_numeric['nuclei'] = pd.to_numeric(win_data_numeric['nuclei'], errors='coerce')  # Convert 'nuclei' column to numeric

win_empty_data = win_data_numeric[win_data_numeric['nuclei'].isnull()]
print('\nThese are the missing data:\n', win_empty_data)

print("\nThese are the data types:")
print(win_data_numeric.dtypes)  # Print the data types

# Comment out the replacement process using SimpleImputer
# imputer = SimpleImputer(strategy='mean')
# win_data_numeric['nuclei'] = imputer.fit_transform(win_data_numeric[['nuclei']])

# Overview of dataset statistics for all columns
print("\nDataset Statistics:")
print(win_data_numeric.describe(include='all'))
