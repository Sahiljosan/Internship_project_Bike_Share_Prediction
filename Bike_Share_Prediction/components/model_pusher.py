import os,sys
import pandas as pd
import numpy as np

from Bike_Share_Prediction.entity import config_entity,artifact_entity
from Bike_Share_Prediction import utils
from Bike_Share_Prediction.utils import load_object,save_object
from Bike_Share_Prediction.config import TARGET_COLUMN
from Bike_Share_Prediction.exception import BikeShareException
from Bike_Share_Prediction.predictor import ModelResolver
from Bike_Share_Prediction.logger import logging
from Bike_Share_Prediction.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ModelPusherArtifact
from Bike_Share_Prediction.entity.config_entity import ModelPusherConfig
from Bike_Share_Prediction.predictor import ModelResolver


class ModelPusher:
    def __init__(self,
                 model_pusher_config:ModelPusherConfig,
                 data_transformation_artifact:DataTransformationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact) -> None:
        try:
            logging.info(f"{'*'*20}Model Pusher{'*'*20}")
            self.model_pusher_config = model_pusher_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver(model_registry= self.model_pusher_config.saved_model_dir)

        except Exception as e:
            raise BikeShareException(e,sys)
        


    def initiate_model_pusher(self,)-> ModelPusherArtifact:
        try:
            # Here we will load our model and target encoder data
            # We have already compared previous model and new model so its time to initiate model pusher for deployment

            logging.info("Loading Transformer model and target encoder")
            transformer = load_object(file_path= self.data_transformation_artifact.transform_object_path)
            model = load_object(file_path= self.model_trainer_artifact.model_path)
            target_encoder = load_object(file_path= self.data_transformation_artifact.target_encoder_path)

            # Model Pusher dir
            logging.info("Saving model into model pusher directory")
            save_object(file_path= self.model_pusher_config.pusher_transformer_path,obj= transformer)
            save_object(file_path = self.model_pusher_config.pusher_model_path,obj= model)
            save_object(file_path=self.model_pusher_config.pusher_target_encoder_path,obj = target_encoder)

            # saved model dir (best model after comparision)
            logging.info(f"Saving model in saved model dir")
            transformer_path = self.model_resolver.get_latest_save_transform_path()
            model_path = self.model_resolver.get_latest_save_model_path()
            target_encoder_path = self.model_resolver.get_latest_save_target_encoder_path()

            save_object(file_path= transformer_path, obj = transformer)
            save_object(file_path= model_path, obj = model)
            save_object(file_path= target_encoder_path, obj = target_encoder)

            model_pusher_artifact = ModelPusherArtifact(pusher_model_dir= self.model_pusher_config.pusher_model_dir,
                                                        saved_model_dir= self.model_pusher_config.saved_model_dir)
            

            logging.info(f"Model Pusher Artifact: {model_pusher_artifact}")
            return model_pusher_artifact


        except Exception as e:
            raise BikeShareException(e,sys)