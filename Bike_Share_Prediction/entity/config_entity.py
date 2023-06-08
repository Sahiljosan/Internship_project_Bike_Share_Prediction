import os,sys
from Bike_Share_Prediction.logger import logging
from Bike_Share_Prediction.exception import BikeShareException
from datetime import datetime


FILE_NAME = "bike_sharing_data.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
TRANSFORMER_OBJECT_FILE_NAME = "transformer.pkl"
TARGET_ENCODER_OBJECT_FILE_NAME = "target_encoder.pkl"
MODEL_FILE_NAME = "model.pkl"



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
        

class DataValidationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir,"data_validation")
        self.report_file_path = os.path.join(self.data_validation_dir, "report.yaml") # report can be created in any format yaml, json, csv
        self.missing_threshold:float = 0.2
        self.base_file_path = os.path.join(r"G:\Udemy\DATA SCIENCE ineuron\VS Code\Internship_project_Bike_Share_Prediction\Data\Bike_Share_Day 2.csv")


class DataTransformationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig) -> None:
        self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir,"data_transformation") 
        self.transform_object_path = os.path.join(self.data_transformation_dir,"transformer", TRANSFORMER_OBJECT_FILE_NAME ) 
        # Calling train data and replacing the extension csv with npz
        self.transform_train_path = os.path.join(self.data_transformation_dir, "transformed" ,TRAIN_FILE_NAME.replace("csv","npz")) 
        self.transform_test_path = os.path.join(self.data_transformation_dir, "transformed" ,TEST_FILE_NAME.replace("csv","npz")) 
       
        self.target_encoder_path = os.path.join(self.data_transformation_dir, "target_encoder" ,TARGET_ENCODER_OBJECT_FILE_NAME) 



class ModelTrainingConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig) -> None:
        self.model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir,"model_trainer")
        self.model_path = os.path.join(self.model_trainer_dir,"model",MODEL_FILE_NAME)
        #set the threshold for the accuracy, if the accuracy is below 70% then model will not accept
        self.expected_accuracy = 0.7
        # set the threshold to check the overfitting
        self.overfitting_threshold = 0.3


# Model Evaluation
class ModelEvaluationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig) -> None:
        self.change_threshold = 0.01


# Model Pusher
class ModelPusherConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig) -> None:
        self.model_pusher_dir = os.path.join(training_pipeline_config.artifact_dir,"model_pusher_file")
        self.saved_model_dir = os.path.join("saved_models")
        self.pusher_model_dir = os.path.join(self.model_pusher_dir,"saved_models")
        self.pusher_model_path = os.path.join(self.pusher_model_dir,MODEL_FILE_NAME)
        self.pusher_transformer_path = os.path.join(self.pusher_model_dir,TRANSFORMER_OBJECT_FILE_NAME)
        self.pusher_target_encoder_path = os.path.join(self.pusher_model_dir,TARGET_ENCODER_OBJECT_FILE_NAME)

        


        



