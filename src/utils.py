import os
import sys
import numpy as np
import pandas as pd
import pickle
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from mlflow.models import infer_signature
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
from sklearn.model_selection import GridSearchCV


# Set tracking URI before run
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

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
    
def hyperparameter_tuning(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3,
                               n_jobs=-1, verbose=2, scoring="neg_mean_squared_error")
    grid_search.fit(X_train, y_train)
    return grid_search
    
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    """
    Evaluate the model performance using R2 score and return the best model.
    """
    try:
        signature = infer_signature(X_train, y_train)
        report = {}
        logging.info("Starting model hyperparameter tuning and evaluation")
        print("Starting model hyperparameter tuning and evaluation")
        for model_name, (model, param_grid) in models.items():
            with mlflow.start_run(run_name=model_name):
                # Hyperparameter tuning
                grid_search = hyperparameter_tuning(model, param_grid, X_train, y_train)
                best_model = grid_search.best_estimator_

                # Evaluate
                y_pred = best_model.predict(X_test)
                r2_square = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                report[model_name] = { 'r2_square': r2_square, 
                                   'mae': mae, 
                                   'mse': mse, 
                                   'rmse': rmse }
                
                # Log model name, parameters, and metrics
                mlflow.log_param("model", model_name)
                for param_name, param_val in grid_search.best_params_.items():
                    mlflow.log_param(param_name, param_val)
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2_score", r2_square)
                
                # Log model
                tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(best_model, "model", registered_model_name=f"Best_{model_name}_Model")
                else:
                    mlflow.sklearn.log_model(best_model, "model", signature=signature)
        
                print(f"{model_name} - Best Hyperparameters: {grid_search.best_params_}")
                print(f"{model_name} - Mean Squared Error: {mse}")
                print(f"{model_name} - Mean Absolute Error: {mae}")
                print(f"{model_name} - Root Mean Squared Error: {rmse}")
                print(f"{model_name} - R2 Score: {r2_square}")
                
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