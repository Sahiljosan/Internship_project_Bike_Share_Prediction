import pandas as pd
import numpy as np
import os
import sys
import yaml
import dill

from Bike_Share_Prediction.logger import logging
from Bike_Share_Prediction.exception import BikeShareException
from Bike_Share_Prediction.config import mongo_client

def get_collection_as_dataframe(database_name:str, collection_name:str)->pd.DataFrame:
    """
    Description: This function return collection as dataframe
    =========================================================
    Params:
    database_name: BIKE_RENTAL
    collection_name: BIKE_SHARING_DAILY
    =========================================================
    return Pandas dataframe of a collection
    """
    try:
        logging.info

    except Exception as e:
        raise BikeShareException(e,sys)
    
       
