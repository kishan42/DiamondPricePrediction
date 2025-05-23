import os
import sys
import numpy as np
import pandas as pd
import pickle
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def save_object(obj, file_path):
    """
    Save the object to a file using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
        logging.info(f"Object saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving object: {str(e)}")
        raise CustomException(e, sys)
    
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    """
    Evaluate the model performance using R2 score and return the best model.
    """
    try:
        report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2_square = r2_score(y_test, y_pred)
            report[model_name] = r2_square
        return report
    except Exception as e:
        logging.error(f"Error evaluating models: {str(e)}")
        raise CustomException(e, sys)
    
def load_object(file_path):
    """
    Load the object from a file using pickle.
    """
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        logging.error(f"Error loading object: {str(e)}")
        raise CustomException(e, sys)