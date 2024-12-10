import pandas as pd
from pycaret.classification import setup, compare_models

# Load the Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

# Handle missing values (simplified for demonstration)
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Select relevant features and target
data = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']]
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# PyCaret setup
clf_setup = setup(data=data, target='Survived', html=False, verbose=False)

# Compare 16 models
best_model = compare_models(n_select=16)

# Output the best models
print(best_model)
