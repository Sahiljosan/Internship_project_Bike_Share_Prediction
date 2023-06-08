
import os,sys
import pandas as pd
import numpy as np

from Bike_Share_Prediction.entity import config_entity,artifact_entity
from Bike_Share_Prediction.config import TARGET_COLUMN,NOT_REQUIRED_COL,INSTANT
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from Bike_Share_Prediction.exception import BikeShareException
from Bike_Share_Prediction.logger import logging
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from Bike_Share_Prediction import utils

# Here we will do
# Missing Values Inpute
# Outliers handling
# Inbalanced data handling
# Convert categorical data into numerical data


class DataTansformation:
    def __init__(self,data_transformation_config:config_entity.DataTransformationConfig,
                 data_ingestion_artifact:artifact_entity.DataIngestionArtifact) -> None:
        
        try:
            logging.info("***********************Data Transformation***********************")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact

        except Exception as e:
            raise BikeShareException(e,sys)
        


    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        try:
            simple_imputer = SimpleImputer(strategy = "constant", fill_value=0)
            #robusst_scaler handles the outliers
            robust_scaler = RobustScaler()

            pipeline = Pipeline(steps = [
                ('Imputer', simple_imputer),
                ('RobustScaler',robust_scaler)
            ])

            return pipeline

        except Exception as e:
            raise BikeShareException(e,sys)
        

    def initiate_data_transformation(self,)->artifact_entity.DataTransformationArtifact:
        """
        ======================================================================================
        We are going to read the data that is available in artifact/dataset/train.csv
        ======================================================================================
        """
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            logging.info("we are droping the target column to separate independent features")
            # we have defined the TARGET_COLUMN in config file
            input_features_train_df = train_df.drop(TARGET_COLUMN,axis=1)
            input_features_test_df = test_df.drop(TARGET_COLUMN,axis=1)

            logging.info("Here we are diffining the dependent feature")
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]

            label_encoder = LabelEncoder()

            # squeeze() function is used to remove single-dimensional entries from the shape of an array.
            target_feature_train_arr = target_feature_train_df.squeeze()
            target_feature_test_arr = target_feature_test_df.squeeze()

            logging.info("Transforming the features whose dtype is object through label encoder")
            for col in input_features_train_df.columns:
                if input_features_train_df[col].dtypes == "O":
                    # if the dtype of feature is object then transform it through label encoder
                    input_features_train_df[col] = label_encoder.fit_transform(input_features_train_df[col])
                    input_features_test_df[col] = label_encoder.fit_transform(input_features_test_df[col])

                else:
                    # if the dtype of feature is numerical then keep it as it is
                    input_features_train_df[col] = input_features_train_df[col]
                    input_features_test_df[col] = input_features_test_df[col]


            transformation_Pipeline = DataTansformation.get_data_transformer_object()
            transformation_Pipeline.fit(input_features_train_df)

            input_features_train_arr = transformation_Pipeline.transform(input_features_train_df)
            input_features_test_arr = transformation_Pipeline.transform(input_features_test_df)

            train_arr = np.c_[input_features_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_features_test_arr,target_feature_test_arr]

            # we are doing this because our data is asking output in array format
            # Right now our data is in str format
            utils.save_numpy_array_data(file_path= self.data_transformation_config.transform_train_path,
                                        array= train_arr)
            
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transform_test_path,
                                        array=test_arr)
            
            utils.save_object(file_path= self.data_transformation_config.transform_object_path,obj= transformation_Pipeline)

            utils.save_object(file_path= self.data_transformation_config.target_encoder_path,obj= label_encoder)



            # Saving all the above code in artifact
            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path = self.data_transformation_config.transform_object_path,
                transform_train_path = self.data_transformation_config.transform_train_path,
                transform_test_path = self.data_transformation_config.transform_test_path,
                target_encoder_path = self.data_transformation_config.target_encoder_path

            )

            return data_transformation_artifact

                    
        except Exception as e:
            raise BikeShareException(e,sys)
        
        
