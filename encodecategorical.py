import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load temperature data into a dataframe
temperature_data = pd.read_csv("temperature.csv")
print(temperature_data)

# Check if 'Temperature' column exists in the dataframe
if 'Temperature' in temperature_data.columns:
    # Convert temperature categories into dummy variables
    temperature_data = pd.get_dummies(temperature_data, columns=['Temperature'], prefix='temp')
    print(temperature_data.head(5))

    # Extract independent variables
    indep_variables = temperature_data.iloc[:, :-1]  # Exclude the last column (dependent variable)
    print('The extracted independent data:')
    print(indep_variables.shape)
    print(indep_variables.tail(5))
    print('----First the last 5 rows of extracted independent data-------')

    # Extract dependent variable
    depent_variables = temperature_data.iloc[:, -1]  # Only the last column (dependent variable)
    print('The extracted dependent data:')
    print(depent_variables.shape)
    print(depent_variables.tail(5))
    print('----First the last 5 rows of extracted dependent data-------')

    # Encode temperature labels
    encoder = LabelEncoder()
    temperature_data['encode_temp_label'] = encoder.fit_transform(temperature_data['Temperature'])
    print(temperature_data.head(5))

else:
    print("Error: 'Temperature' column not found in the dataframe.")

# Load iris data into a dataframe
header_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'iris_class']
iris_data = pd.read_csv("iris_data.csv", sep=',\s*', skipinitialspace=False, engine='python', names=header_names)
print(iris_data.head(5))

# Encode iris class labels
encoder = LabelEncoder()
iris_data['encode_class_label'] = encoder.fit_transform(iris_data['iris_class'])
print(iris_data.head(3))
print(iris_data.tail(3))

# Declare order dictionary for temperature
temp_declaration = {"Cold": 1, "Warm": 2, "Hot": 3, "Very hot": 4}

# Encode temperature using ordinal encoding
if 'Temperature' in iris_data.columns:
    iris_data['ordinal_encoded'] = iris_data['Temperature'].map(temp_declaration)
    print(iris_data.head(5))
else:
    print("Error: 'Temperature' column not found in the dataframe.")

# Convert temperature categories into dummy variables
iris_data = pd.get_dummies(iris_data, columns=['Temperature'], prefix='   temp')
print(iris_data.head(5))
