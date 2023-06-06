from Bike_Share_Prediction.logger import logging
from Bike_Share_Prediction.exception import BikeShareException
import os,sys

from Bike_Share_Prediction.utils import get_collection_as_dataframe
from Bike_Share_Prediction.components.data_ingestion import DataIngestion
from Bike_Share_Prediction.entity import config_entity
from Bike_Share_Prediction.entity import artifact_entity


# def test_logger_and_exception():
#     try:
#         logging.info("Starting the test logger and exception")
#         result = 3/0
#         print(result)
#         logging.info("Ending point of the logger and exception")

#     except Exception as e:
#         logging.debug(str(e))
#         raise BikeShareException(e,sys)
    

# if __name__ == "__main__":
#     try:
#         test_logger_and_exception()

#     except Exception as e:
#         print(e)

if __name__ == "__main__":
    try:
        # get_collection_as_dataframe(database_name="BIKE_RENTAL", collection_name="BIKE_SHARING_DAILY")

        training_pipeline_config = config_entity.TrainingPipelineConfig()

        # data_ingestion
        data_ingestion_config = config_entity.DataIngestionConfig(training_pipeline_config = training_pipeline_config)
        print(data_ingestion_config.to_dict())

        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()


    except Exception as e:
        raise BikeShareException(e,sys)
