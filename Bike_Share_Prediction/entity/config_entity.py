import os,sys
from Bike_Share_Prediction.logger import logging
from Bike_Share_Prediction.exception import BikeShareException
from datetime import datetime


FILE_NAME = "bike_sharing_data.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"

class TrainingPipelineConfig:
    def __init__(self):      
        try:
            self.artifact_dir = os.path.join(os.getcwd(),"artifact",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")
        except Exception as e:
            raise BikeShareException(e,sys)
        

class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            self.database_name = "BIKE_RENTAL"
            self.collection_name = "BIKE_SHARING_DAILY"
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir,"data_ingestion") # we are creating one folder of name "data_ingestion" in artifact directory
            self.feature_store_file_path = os.path.join(self.data_ingestion_dir,"feature_store", FILE_NAME)
            self.train_file_path = os.path.join(self.data_ingestion_dir,"dataset",TRAIN_FILE_NAME)
            self.test_file_path = os.path.join(self.data_ingestion_dir,"dataset",TEST_FILE_NAME)
            self.test_size = 0.2

        except Exception as e:
            raise BikeShareException(e,sys)
        
# Convert data into dict
    def to_dict(self)->dict:
        try:
            return self.__dict__
        except Exception as e:
            raise BikeShareException(e,sys)

