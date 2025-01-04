# Titanic Survival Prediction Using PyCaret

A machine learning project using PyCaret to predict passenger survival on the Titanic, following the CRISP-DM methodology.

## 1. Business Understanding

### Project Goal
- Predict passenger survival on the Titanic using machine learning models
- Identify key factors that influenced survival probability
- Build a reliable prediction model with high accuracy

### Success Metrics
- Model Accuracy > 80%
- AUC-ROC Score > 0.85
- F1-Score > 0.75

### Business Value
- Understanding historical passenger survival patterns
- Insights into factors affecting survival rates
- Development of a predictive model for similar scenarios

## 2. Data Understanding

### Dataset Overview
- Source: Titanic Dataset (Kaggle)
- Features: 12 variables including:
  - PassengerID
  - Survival Status
  - Passenger Class
  - Name, Sex, Age
  - SibSp, Parch
  - Ticket, Fare
  - Cabin, Embarked

### Initial Analysis
```python
print("Dataset Shape:", data.shape)
print("Missing Values:\n", data.isnull().sum())
print("Feature Statistics:\n", data.describe())
```

### Key Insights
- 891 passengers in the dataset
- 38.4% survival rate
- Missing values in Age (177), Cabin (687), Embarked (2)
- Significant class imbalance in survival

## 3. Data Preparation

### Feature Engineering
```python
# Created new features
data['Family_Size'] = data['SibSp'] + data['Parch'] + 1
data['Is_Alone'] = (data['Family_Size'] == 1).astype(int)
```

### Data Processing
1. Missing Value Treatment
   - Age: Median imputation
   - Embarked: Mode imputation
   - Fare: Median imputation

2. Feature Creation
   - Family_Size
   - Is_Alone
   - Fare_Bin (quartile-based)
   - Age_Bin (custom ranges)

3. Feature Selection
   - Dropped: PassengerId, Name, Ticket, Cabin
   - Kept: All engineered features and remaining original features

## 4. Modeling

### Model Setup
```python
exp = setup(
    data=processed_data,
    target='Survived',
    numeric_features=['Age', 'Fare', 'Family_Size'],
    categorical_features=['Sex', 'Embarked', 'Pclass', 'Fare_Bin', 'Age_Bin'],
    normalize=True,
    transformation=True,
    ignore_features=['SibSp', 'Parch'],
    session_id=42
)
```

### Models Tested
1. Gradient Boosting Classifier
2. Light Gradient Boosting Machine
3. Ada Boost Classifier
4. Random Forest
5. Logistic Regression
6. Decision Trees
7. SVM
8. KNN

## 5. Evaluation

### Best Model Performance
- Gradient Boosting Classifier:
  - Accuracy: 82.02%
  - AUC: 0.8696
  - Recall: 0.7071
  - Precision: 0.8012
  - F1 Score: 0.7511

### Visualization
- Confusion Matrix
- AUC-ROC Curve
- Feature Importance Plot
- Model Performance Comparison

## 6. Deployment

### Model Export
```python
# Save model
prediction_function = prepare_deployment(best_model)
```

### Sample Prediction
```python
# Example usage
sample_passenger = processed_data.iloc[[0]].copy()
prediction = prediction_function(sample_passenger)
```

## Requirements
```
pandas
numpy
pycaret
matplotlib
seaborn
scikit-learn
```

## Installation & Usage
1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Run the project:
```python
python titanic_prediction.py
```

## File Structure
```
project/
├── README.md
├── requirements.txt
├── notebooks/
│   └── titanic_analysis.ipynb
├── src/
│   ├── data_preparation.py
│   ├── modeling.py
│   └── evaluation.py
└── models/
    └── best_model.pkl
```

## Future Improvements
- [ ] Feature selection optimization
- [ ] Hyperparameter tuning
- [ ] Ensemble model exploration
- [ ] Web interface development
- [ ] Real-time prediction API

## Contact
For questions and feedback:
- Email: [your-email]
- GitHub: [your-github]

## License
This project is licensed under the MIT License - see the LICENSE file for details.
