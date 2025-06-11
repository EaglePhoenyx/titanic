"""
Train the Titanic model.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def train_model(X_train,y_train):
    """ 
    Initiate the model and train it on the Titanic dataset.""
    Returns the train model
    """

    lin = LogisticRegression(max_iter=1000, random_state=42)
    lin.fit(X_train, y_train)
    return lin

def evaluate_model(model,X_test,y_test):
    """
    Input : Model, X_test,y_test
    Output: return the model score against X_test and y_test
    """
    metric = model.score(X_test, y_test)
    metric = f'{metric:.2f}'
    print(f"Model has a score of {metric}")

def optimize_model(model,X_train,y_train):
    param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    print("Best parameters:", grid_search.best_params_, "Best score:", grid_search.best_score_)
    best_model = grid_search.best_estimator_
    return best_model

