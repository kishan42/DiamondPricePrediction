import numpy as np
import pandas as pd
import os
## Model Training
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model

from dataclasses import dataclass
import sys
import os


@dataclass
class ModelTrainerConfig:
    """Model Trainer Configuration Class"""
    trained_moddel_file_path: str = os.path.join("artifacts", "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            logging.info("Training model")
            
            models = {
                "LinearRegression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "ElasticNet": ElasticNet(),
                "DecisionTreeRegressor": DecisionTreeRegressor()
            }
            
            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            logging.info("Model report generated")
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            logging.info(f"Best model found: {best_model_name[0]} with score: {best_model_score}")
            
            save_object(best_model, self.model_trainer_config.trained_moddel_file_path)
            logging.info(f"Model saved to {self.model_trainer_config.trained_moddel_file_path}")
            
            
        except Exception as e:
            logging.error("Error occurred during model training")
            raise CustomException(e, sys)
            
            
            
            
            
            
        