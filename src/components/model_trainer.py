import numpy as np
import pandas as pd
import os
## Model Training
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
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
            
            # Models and their respective parameter grids
            models = {
                'LinearRegression': (LinearRegression(), {}),
                'Lasso': (Lasso(), {
                    'alpha': [0.01, 0.1, 1.0]
                }),
                'Ridge': (Ridge(), {
                    'alpha': [0.01, 0.1, 1.0]
                }),
                'ElasticNet': (ElasticNet(), {
                    'alpha': [0.01, 0.1, 1.0],
                    'l1_ratio': [0.1, 0.5, 0.9]
                }),
                'RandomForestRegressor': (RandomForestRegressor(), {
                    'n_estimators': [100, 200],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }),
                'DecisionTreeRegressor': (DecisionTreeRegressor(), {
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                })
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
            
            
            
            
            
            
        