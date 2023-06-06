import os,sys
import pandas as pd
import numpy as np

from Bike_Share_Prediction.entity import config_entity,artifact_entity
from Bike_Share_Prediction import utils
from Bike_Share_Prediction.config import TARGET_COLUMN
from Bike_Share_Prediction.exception import BikeShareException
from Bike_Share_Prediction.logger import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Define the model
# Train the model
# Set Accuracy
# check the model is overfitted or underfitted

class ModelTrainer:
    # define constructor
    def __init__(self,model_trainer_config:config_entity.ModelTrainingConfig,
                 data_transformation_artifact:artifact_entity.DataTransformationArtifact) -> None:
        try:
            logging.info("*******************Model Trainer*******************")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise BikeShareException(e,sys)
        
    
    # Define the model 
    def train_model(self, x,y):
        try:
            lr = LinearRegression()
            lr.fit(x,y)
            return lr

        except Exception as e:
            raise BikeShareException(e,sys) 
        

    
    def initiate_model_trainer(self,)->artifact_entity.ModelTrainerArtifact:
        try:
            # same as jupyter notebook, we are loading the data (from utils file)
            train_arr = utils.load_numpy_array_data(file_path= self.data_transformation_artifact.transform_train_path)
            test_arr = utils.load_numpy_array_data(file_path= self.data_transformation_artifact.transform_test_path)

            logging.info("dividing the data into x_train,y_train,x_test,y_test")
            x_train,y_train = train_arr[:,:-1],train_arr[:,-1]
            x_test,y_test = test_arr[:,:-1],test_arr[:,-1]

            logging.info("Building Model")
            model = self.train_model(x = x_train,y = y_train)

            logging.info("Store r2_score for train data")
            yhat_train = model.predict(x_train)
            r2_train_score = r2_score(y_true=y_train, y_pred= yhat_train)
            logging.info("Store r2_score for test data")
            yhat_test = model.predict(x_test)
            r2_test_score = r2_score(y_true=y_test, y_pred= yhat_test)

            logging.info("setting the threshold")
            if r2_test_score < self.model_trainer_config.expected_accuracy:
                raise Exception(f"Model is not good as it is not able to \
                                give expected accuracy:{self.model_trainer_config.expected_accuracy}: \
                                model actual score: {r2_test_score}")
            
            diff = abs(r2_train_score - r2_test_score)

            logging.info("Checking the overfitting")
            if diff > self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train model and test score diff : {diff} is more then overfitting threshold {self.model_trainer_config.overfitting_threshold}")
            
            utils.save_object(file_path= self.model_trainer_config.model_path,obj= model) # the "model" which we have defined in config_entity.py file

            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(model_path= self.model_trainer_config.model_path,
                                                                          r2_train_score= r2_train_score,r2_test_score= r2_test_score)
            
            return model_trainer_artifact
            

        except Exception as e:
            raise BikeShareException(e,sys) 

