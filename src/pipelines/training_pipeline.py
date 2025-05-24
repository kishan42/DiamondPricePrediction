import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    try:
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        print("Data ingestion completed successfully.",train_data_path, test_data_path)
        logging.info(f"Train data path: {train_data_path}")
        logging.info(f"Test data path: {test_data_path}")
        
        data_transformation = DataTransformation()
        print("Data transformation started.")
        logging.info("Data transformation started.")
        train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_file_path=train_data_path, test_file_path=test_data_path)
        print("Data transformation completed successfully.")
        
        print("Model training started.")
        logging.info("Model training started.")
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_trainer(train_array=train_arr, test_array=test_arr)
        print("Model training completed successfully.")
        
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")