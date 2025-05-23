from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

## data transformation configuration
@dataclass
class DataTransformationConfig:
    """Data Transformation Configuration Class"""
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

## data Transformationconfig class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            logging.info("Data Transformation initiated")
            # Define the numerical and categorical columns
            numerical_columns = ["carat", "depth", "table", "x", "y", "z"]
            categorical_columns = ["cut", "color", "clarity"]
            
            
            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info("pipeline initiated")
            
            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
            
                ]
            
            )
            
            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                ('scaler',StandardScaler())
                ]
            
            )
            
            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_columns),
            ('cat_pipeline',cat_pipeline,categorical_columns)
            ])
            
            return preprocessor
            
        except Exception as e:
            logging.info("Exception occurred in data transformation")
            raise CustomException(e, sys)

          
    def initiate_data_transformation(self, train_file_path, test_file_path):
        try:
            # Read the training and testing data
            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)
            
            logging.info("Read train and test data completed")
            logging.info(f"train df head: {train_df.head().to_string()}")
            logging.info(f"test df head: {test_df.head().to_string()}")

            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj = self.get_data_transformer_object()
            logging.info("Preprocessing object obtained")
            
            target_column_name = "price"
            drop_columns = [target_column_name, "id"]
            
            ## features into independent and dependent
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            
            ## apply the transformation
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            logging.info("Applying preprocessing object on training dataframe and testing dataframe")
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info("Data transformation completed")
            
            
            save_object(
                obj=preprocessing_obj,
                file_path=self.data_transformation_config.preprocessor_obj_file_path
            )
            
            logging.info("Preprocessor pickle file saved")
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            logging.info("Exception occurred in data transformation")
            raise CustomException(e, sys)
            
            