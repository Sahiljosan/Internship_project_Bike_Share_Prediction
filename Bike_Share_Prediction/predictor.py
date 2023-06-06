import os,sys
import pandas as pd
import numpy as np
from Bike_Share_Prediction.logger import logging
from Bike_Share_Prediction.exception import BikeShareException
from Bike_Share_Prediction.logger import logging
from Bike_Share_Prediction.entity import config_entity,artifact_entity
from typing import Optional

from Bike_Share_Prediction.entity.config_entity import MODEL_FILE_NAME, TARGET_ENCODER_OBJECT_FILE_NAME, TRANSFORMER_OBJECT_FILE_NAME

"""
==========================================================================================
When we are running data multiple times then what will happen ?
1. One folder will be created with name "saved_models"
2. if we are running pipeline 1st time, then 1st folder will be created '0'
3. if we are running 2nd time, then folder will be created with name '1' and it will contain folders of transformer, target_encode, model
4. if we are running 3rd time, then folder will be created with name '2' 
and so on
==========================================================================================
"""

class ModelResolver:
    # define constructor
    def __init__(self, model_registry:str = "saved_models",
                 transformer_dir_name = "transformer",
                 target_encoder_dir_name = "target_encoder",
                 model_dir_name = "model") -> None:
        self.model_registry = model_registry
        os.makedirs(self.model_registry, exist_ok= True)
        self.transformer_dir_name = transformer_dir_name
        self.target_encoder_dir_name = target_encoder_dir_name
        self.model_dir_name = model_dir_name

# 1
    def get_latest_dir_path(self)->Optional[str]:
        try:
            dir_name = os.listdir(self.model_registry)
            if len(dir_name) == 0:
                return None
            

            dir_name = list(map(int, dir_name))
            latest_dir_name = max(dir_name)
            return os.path.join(self.model_registry, f"{latest_dir_name}")


        except Exception as e:
            raise (e,sys)
        

# 2
    def get_latest_model_path(self):
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f"Model is not available")
            
            return os.path.join(latest_dir, self.model_dir_name, MODEL_FILE_NAME)

        except Exception as e:
            raise BikeShareException(e,sys)
        

# 3
    def get_latest_transformer_path(self):
         """
         ========================================================================================
         Here 1st we are checking is the folder is available or not in the latest directory.
         If not available then raise exception that transform data is not available
         and after that returning the directory with old data.
         ========================================================================================
         """
          
         try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f"Transformer is not available")
            
            return os.path.join(latest_dir, self.transformer_dir_name, TRANSFORMER_OBJECT_FILE_NAME)

         except Exception as e:
            raise BikeShareException(e,sys)

# 4
    def get_latest_target_encoder_path(self):
         
         try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f"Target encoder data is not available")
            
            
            return os.path.join(latest_dir,self.target_encoder_dir_name, TARGET_ENCODER_OBJECT_FILE_NAME)

         except Exception as e:
            raise BikeShareException(e,sys)
         

# 5
    def get_latest_save_dir_path(self)-> str:
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                return os.path.join(self.model_registry, f"{0}")
            
            latest_dir_num = int(os.path.basename(self.get_latest_dir_path()))
            return os.path.join(self.model_registry, f"{latest_dir_num + 1}") # add +1 so that it will increase everytime

        except Exception as e:
            raise BikeShareException(e,sys)
        


# 6 
    def get_latest_save_model_path(self):
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir, self.model_dir_name, MODEL_FILE_NAME) # This will save in pkl file format

        except Exception as e:
            raise BikeShareException(e,sys)
        

# 7
    def get_latest_save_transform_path(self):
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir, self.transformer_dir_name,TRANSFORMER_OBJECT_FILE_NAME) # this will save in transform.pkl format
        
        
        except Exception as e:
            raise BikeShareException(e,sys)
        

# 8 
    def get_latest_save_target_encoder_path(self):
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir, self.target_encoder_dir_name, TARGET_ENCODER_OBJECT_FILE_NAME) # this will save in encoder.pkl format
        
        except Exception as e:
            raise BikeShareException(e,sys)
