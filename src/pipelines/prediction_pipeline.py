import sys
import os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object



class CustomData:
    def __init__(self, carat, depth, table, x, y, z, cut, color, clarity):
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def get_data_as_dataframe(self):
        import pandas as pd
        custom_data_input_dict = {
            "carat": [self.carat],
            "depth": [self.depth],
            "table": [self.table],
            "x": [self.x],
            "y": [self.y],
            "z": [self.z],
            "cut": [self.cut],
            "color": [self.color],
            "clarity": [self.clarity],
        }
        return pd.DataFrame(custom_data_input_dict)

        
        
class PredictPipeline:
    def __init__(self):
        pass

        
    def predict(self, features: list) -> float:
        try:
            preprocessor_path = os.path.join('artifacts', "preprocessor.pkl")
            model_path = os.path.join('artifacts', "model.pkl")
            
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            
            data_scale = preprocessor.transform(features)   
            logging.info(f"Scaled data: {data_scale}")
            ## Convert the scaled data to a DataFrame  
            pred = model.predict(data_scale)
            logging.info(f"Prediction: {pred}")
            
            return pred
            
            
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            raise CustomException(e, sys) from e
