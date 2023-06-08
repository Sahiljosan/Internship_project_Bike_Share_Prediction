import os,sys
import pandas as pd
import numpy as np

from Bike_Share_Prediction.logger import logging
from Bike_Share_Prediction.exception import BikeShareException
from Bike_Share_Prediction.predictor import ModelResolver
from Bike_Share_Prediction.utils import load_object
from Bike_Share_Prediction.entity import config_entity,artifact_entity
from typing import Optional
from datetime import datetime

from Bike_Share_Prediction.components.data_ingestion import DataIngestion
from Bike_Share_Prediction.components.data_validation import DataValidation
from Bike_Share_Prediction.components.data_transformation import DataTansformation
from Bike_Share_Prediction.components.Model_Trainer import ModelTrainer
from Bike_Share_Prediction.components.model_evaluation import ModelEvaluation
from Bike_Share_Prediction.components.model_pusher import ModelPusher


def start_training_pipeline():
    try:
        training_pipeline_config = config_entity.TrainingPipelineConfig()
        
        # Data Ingestion
        data_ingestion_config = config_entity.DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        print(data_ingestion_config.to_dict())
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

        # Data Validation
        """
        In data validation we are going to use the file that is available in artifact folder
        """
        data_validation_config = config_entity.DataValidationConfig(training_pipeline_config= training_pipeline_config)
        data_validation = DataValidation(data_validation_config= data_validation_config,
                                         data_ingestion_artifact=data_ingestion_artifact)
        
        data_validation_artifact = data_validation.initiate_data_validation()

        # Data Transformation
        data_transformation_config = config_entity.DataTransformationConfig(training_pipeline_config=training_pipeline_config)
        data_transformation = DataTansformation(data_transformation_config = data_transformation_config,
                                                 data_ingestion_artifact = data_ingestion_artifact)
        
        data_transformation_artifact = data_transformation.initiate_data_transformation()


        # Model Trainer
        model_trainer_config = config_entity.ModelTrainingConfig(training_pipeline_config=training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config = model_trainer_config,
                                     data_transformation_artifact = data_transformation_artifact)
        
        model_trainer_artifact = model_trainer.initiate_model_trainer()

        # Model Evaluation
        model_eval_config = config_entity.ModelEvaluationConfig(training_pipeline_config= training_pipeline_config)
        model_eval = ModelEvaluation(model_eval_config = model_eval_config, 
                                     data_ingestion_artifact = data_ingestion_artifact,
                                     data_transformation_artifact = data_transformation_artifact,
                                     model_trainer_artifact = model_trainer_artifact)
        
        model_eval_artifact = model_eval.initiate_model_evaluation()
    
        # Model Pusher
        model_pusher_config = config_entity.ModelPusherConfig(training_pipeline_config=training_pipeline_config)
        model_pusher = ModelPusher(model_pusher_config = model_pusher_config,
                                   data_transformation_artifact = data_transformation_artifact,
                                   model_trainer_artifact = model_trainer_artifact)
        
        model_pusher_artifact = model_pusher.initiate_model_pusher()

    except Exception as e:
        raise BikeShareException(e,sys)