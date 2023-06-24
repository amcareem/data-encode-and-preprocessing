import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, Normalizer, Binarizer, StandardScaler

# Loading the data into a DataFrame
header_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
my_data = pd.read_csv("venv/pima_indians_diabetes_2.csv", names=header_names)

# Getting the data dimensions
print("Data Dimensions:")
print(my_data.shape)

# Printing the first 5 rows
print("\nFirst 5 Rows:")
print(my_data.head(5))
# Printing the last 5 rows
print("\nLast 5 Rows:")
print(my_data.tail(5))

# Checking the data which has empty variables
empty_data = my_data[my_data.isna().any(axis=1)]
print("\nData with Empty Variables:")
print(empty_data)

# Replacing the empty values with mean using SimpleImputer
# SimpleImputer is a class in the sklearn. impute module that can be used to replace missing values in a dataset
imputer = SimpleImputer(strategy='mean')
imputed_data = pd.DataFrame(imputer.fit_transform(my_data), columns=header_names)

# Data scaling using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = pd.DataFrame(scaler.fit_transform(imputed_data), columns=header_names)

# L1 normalization using Normalizer
normalizer_l1 = Normalizer(norm='l1')
normalized_l1 = pd.DataFrame(normalizer_l1.transform(scaled_data), columns=header_names)

# L2 normalization using Normalizer
normalizer_l2 = Normalizer(norm='l2')
normalized_l2 = pd.DataFrame(normalizer_l2.transform(scaled_data), columns=header_names)

# Binarization using Binarizer
binarizer = Binarizer(threshold=0.5)
binarized_data = pd.DataFrame(binarizer.transform(imputed_data), columns=header_names)

# Standardization using StandardScaler
scaler_standard = StandardScaler()
standardized_data = pd.DataFrame(scaler_standard.fit_transform(imputed_data), columns=header_names)

# Printing all the resultsssssssssssssssssssssssssssssssssssssssssssssss
print("\nScaled Data:")
print(scaled_data.head(5))
print("\nNormalized Data (L1):")
print(normalized_l1.head(5))
print("\nNormalized Data (L2):")
print(normalized_l2.head(5))
print("\nBinarized Data:")
print(binarized_data.head(5))
print("\nStandardized Data:")
print(standardized_data.head(5))