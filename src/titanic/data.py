"""
Load, preprocess, prepare, and save the Titanic dataset.
"""

import pandas as pd
from sklearn.impute import SimpleImputer


def load_data():
    """
    Load the Titanic dataset from a CSV file.
    
    Returns:
        DataFrame: The loaded Titanic dataset.
    """
    pass
       
def clean_data(df):
    """
    clean the Titanic dataset.
    
    Args:
        df (DataFrame): The Titanic dataset.
        
    Returns:
        DataFrame: The preprocessed Titanic dataset.
    """
    # Drop the 3 columns Name, Ticket, Cabin
    df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)
    # Replacing missing values with scikit-learn's SimpleImputer
    # We have missing values in Age and Embarked columns
    imputer = SimpleImputer().set_output(transform="pandas")
    imputer.fit(df[['Age']])
    df[['Age']] = imputer.transform(df[['Age']])
    df["Embarked"].fillna("S", inplace=True)

    return df

def prepare_data(df:pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]: 
    """
    Prepare the Titanic dataset for training.
    
    Args:
        df (DataFrame): The preprocessed Titanic dataset.
        
    Returns:
        tuple: A tuple containing [X,y] the features DataFrame and the target Series.
    """
    pass