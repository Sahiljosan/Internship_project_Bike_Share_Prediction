import pymongo 
import pandas as pd
import json

client = pymongo.MongoClient("mongodb+srv://sahil_josan:samongodbhil5@cluster0.sptya9h.mongodb.net/?retryWrites=true&w=majority")

DATA_FILE_PATH = (r"G:\Udemy\DATA SCIENCE ineuron\VS Code\Internship_project_Bike_Share_Prediction\Data\Bike_Share_Day.csv")

DATABASE_NAME = "BIKE_RENTAL"

COLLECTION_NAME = "BIKE_SHARING_DAILY"

if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and Columns : {df.shape}")

    df.reset_index(drop= True, inplace= True)

    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[0])

    client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)


