# we have created this file for : 
# batch_prediction
# training_pipeline

from Bike_Share_Prediction.pipeline.batch_prediction import start_batch_prediction
from Bike_Share_Prediction.pipeline.training_pipeline import start_training_pipeline

# file_path = r"G:\Udemy\DATA SCIENCE ineuron\VS Code\Internship_project_Bike_Share_Prediction\Data\Bike_Share_Day 2.csv"

if __name__ == "__main__":
    try:
        #output_file = start_batch_prediction(input_file_path= file_path)
        output = start_training_pipeline()
        print(output)

    except Exception as e:
        print(e)

