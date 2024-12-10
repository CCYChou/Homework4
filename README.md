
# Titanic Dataset Model Comparison with PyCaret

This project demonstrates how to use PyCaret to preprocess the Titanic dataset and compare the performance of 16 different machine learning models. The script automates model comparison, enabling easy evaluation of model performance on a binary classification task.

## Features
- Preprocessing of the Titanic dataset:
  - Imputation of missing values.
  - One-hot encoding of categorical variables (`Sex` and `Embarked`).
- Automated comparison of 16 machine learning models using PyCaret.
- Output of ranked models based on key performance metrics.

## Prerequisites
Before running the script, ensure you have the following installed:
- Python 3.8 or higher
- Required Python libraries:
  - `pandas`
  - `pycaret`

## Installation

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd <repository-folder>
```

### Step 2: Set Up the Environment
1. Create and activate a Python environment:
   ```bash
   conda create -n pycaret_env python=3.8 -y
   conda activate pycaret_env
   ```
2. Install the required packages:
   ```bash
   pip install pandas pycaret
   ```

## Usage

1. Run the Python script:
   ```bash
   python compare_titanic_models.py
   ```
2. The script performs the following steps:
   - Loads the Titanic dataset from a public GitHub URL.
   - Preprocesses the data (handles missing values and encodes categorical variables).
   - Sets up PyCaret for classification with the target variable `Survived`.
   - Compares 16 machine learning models and prints their performance metrics.

### Example Output
The script outputs:
1. A ranked list of 16 models based on performance metrics such as **Accuracy**, **AUC**, **Recall**, **Precision**, and **F1** score.
2. The best models’ configurations for further customization.

### Sample Output Snippet
```plaintext
                                    Model  Accuracy     AUC  Recall   Prec.  
rf               Random Forest Classifier    0.8045  0.8625  0.7246  0.7663
gbc          Gradient Boosting Classifier    0.8013  0.8609  0.6701  0.7936
qda       Quadratic Discriminant Analysis    0.7981  0.8428  0.6824  0.7771
...
```

## Dataset Information
The Titanic dataset contains information about passengers and their survival status. Features used in this project:
- **Pclass**: Passenger class (1, 2, 3)
- **Sex**: Gender (converted to binary variables)
- **Age**: Age of the passenger (missing values imputed)
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Fare**: Ticket fare
- **Embarked**: Port of embarkation (converted to binary variables)

The target variable is:
- **Survived**:  
  - `0`: Did not survive  
  - `1`: Survived  

## Results
- The script ranks 16 machine learning models based on their performance metrics.
- Models are evaluated for their ability to classify passengers as survivors or non-survivors.
- The best-performing models are ready for further tuning and deployment.

## License
This project is licensed under the MIT License. Feel free to use and modify the code as needed.

## Acknowledgments
- [DatascienceDojo](https://github.com/datasciencedojo/datasets) for the Titanic dataset.
- [PyCaret](https://pycaret.org/) for simplifying machine learning workflows.
