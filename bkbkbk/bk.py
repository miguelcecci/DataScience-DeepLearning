import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix


# Static paths
SCALER_PATH = '../models/scaler.joblib'

MODEL_FILE_PATH = '../models/model.joblib'

TRAINING_FILE_BASE_PATH = '../data/Abandono_clientes.csv'

TEST_FILE_BASE_PATH = '../data/Abandono_teste.csv'

# Static lists
REQUIRED_TRAINING_COLUMNS = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
       'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
       'Exited']

REQUIRED_PREDICT_COLUMNS = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
       'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']


# numeric columns
NUMERICS = [
    'CreditScore',
    'Age',
    'Tenure',
    'Balance',
    'NumOfProducts',
    'EstimatedSalary',
]

# every possible value for each categorical column
CATEGORICALS = {
    'Geography': ['France', 'Germany', 'Spain'], 
    'Gender': ['Male', 'Female'], 
    'HasCrCard': [0, 1], 
    'IsActiveMember': [0, 1], 
}



def load_training_data() -> pd.core.frame.DataFrame:
    
    """
    Returns the data for training
    
    Parameters:
        None
    
    Returns:
        Pandas DataFrame containing the training data
    """
    
    df = pd.read_csv(TRAINING_FILE_BASE_PATH)
    
    return df

def load_data() -> pd.core.frame.DataFrame:
    
    """
    Returns the data for predicting
    
    Parameters:
        None
    
    Returns:
        Pandas DataFrame containing the predicting data
    """
    
    df = pd.read_csv(TEST_FILE_BASE_PATH, sep=';')
    
    return df

def assert_categoricals(df: pd.core.frame.DataFrame) -> bool:
    
    """Asserts Categorical data to avoid unexpected new categorical values in each column"""
    
    #Asserting categorical values
    for i in CATEGORICALS:
        # Iterating over every distinct value in column to check if it maches the CATEGORICALS dict
        for unique_value in df[i].unique():
            if not unique_value in CATEGORICALS[i]:
                print('{} value for column {}, is not contained in CATEGORICALS dict'
                      .format(unique_value, i))
                return False
    return True

def assert_data(df: pd.core.frame.DataFrame) -> None:
    
    """
    Do Assertions for each column from dataframe
    
    Parameters:
        Training data or Test data, as Pandas DataFrame
        
    Returns:
        None
    """
    
    assertions_pass = True
    
    assertion_pass = assertions_pass and assert_categoricals(df)
    
    # Asserting that df contains all columns from REQUIRED_TRAINING_COLUMNS
    
    for i in REQUIRED_TRAINING_COLUMNS:
        if not i in df.columns:
            assertion_pass = False
            print("Missing Column: {}".format(i))
            

    
    if assertions_pass:
        print('All assertions passed')
    
    return None
            
def treat_categorical_data(data: pd.core.frame.DataFrame, training = False) -> pd.core.frame.DataFrame:
    
    """Returns the Dataframe with treated categorical columns"""
    
    # Avoiding overwriting
    data = data.copy()
    
    assert(assert_categoricals(data))
    
    columns = list(CATEGORICALS.keys())
    
    return pd.get_dummies(data[columns])

def treat_numeric_data(data: pd.core.frame.DataFrame, training = False) -> pd.core.frame.DataFrame:
    
    """docstring"""
    
#     Avoiding overwriting
    data = data.copy()
    
    columns = NUMERICS
    
    if training:
        scaler = MinMaxScaler()
        scaler.fit(data[columns])
        joblib.dump(scaler, SCALER_PATH)
    else:
        scaler = joblib.load(SCALER_PATH)
        
    transformed_data = scaler.transform(data[columns])
    
    scaled_data = pd.DataFrame(transformed_data, columns=data[columns].columns)
    
    return scaled_data


def treat_data(data: pd.core.frame.DataFrame, training=False) -> pd.core.frame.DataFrame:
    
    ""
    
    data = data.copy()
    
    return pd.concat([treat_numeric_data(data, training=training), treat_categorical_data(data, training=training)], axis=1)

def get_training_dataset() -> pd.core.frame.DataFrame:
    
    """Returns training dataset ready for model train"""
    
    # The main idea about the training parameter is to Reset the scaler when we train the model
    data = load_training_data()
    treated = treat_data(data, training=True)
    treated['Labels'] = data['Exited']
    
    return treated.sample(frac=1).reset_index(drop=True)
    
    
def get_test_dataset() -> pd.core.frame.DataFrame:
    
    """Returns test dataset ready for predictions"""
    
    return treat_data(load_data())
    
    
def compute_metrics(y_true, y_pred) -> None:
    
    """Compute metrics of our model"""
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    print('True Positives: {}'.format(tp))
    print('False Positives: {}'.format(fp))
    print('True Negatives: {}'.format(tn))
    print('False Negatives: {}'.format(fn))
    
    print('Recall: {}'.format(recall_score(y_true, y_pred)))
    print('Precision: {}'.format(precision_score(y_true, y_pred)))
    print('Accuracy: {}'.format(accuracy_score(y_true, y_pred)))
    print('F1 Score: {}'.format(f1_score(y_true, y_pred)))
    
    return None
    
def save_model(model) -> None:
      
    """Persist model"""
    
    joblib.dump(model, MODEL_FILE_PATH)
    print("Modelo Salvo em: {}", MODEL_FILE_PATH)
    
    return None

    