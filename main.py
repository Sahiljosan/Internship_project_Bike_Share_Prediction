from Bike_Share_Prediction.logger import logging

import os,sys


def test_logger_and_exception():
    try:
        logging.info("Starting the test logger and exception")
        result = 3/0
        print(result)
        logging.info("Ending point of the logger and exception")

    except Exception as e:
        logging.debug(str(e))
        raise BikeShareException(e,sys)
    

if __name__ == "__main__":
    try:
        test_logger_and_exception()

    except Exception as e:
        print(e)

