"""
Save and load models, preprocessors
"""
import pickle, os

if not os.path.exists("models"):
    os.makedirs("models")

def save_model(model, path: str):
    """
    Save a model to the specified path.
    
    Args:
        model: The model to save.
        path (str): The file path where the model will be saved.
    """
    with open(path , "wb") as f:
        pickle.dump(model, f)

def load_model(path: str):
    """
    Load a model from the specified path.
    
    Args:
        path (str): The file path from which the model will be loaded.
        
    Returns:
        The loaded model.
    """
    with open(path,'rb' ) as f:
        loaded_model = pickle.load(f)
    return loaded_model