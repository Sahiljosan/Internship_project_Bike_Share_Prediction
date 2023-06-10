import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

col1, col2 = st.columns(2)
with col1:
       def add_bg_from_local(image_file):
             
              with open(image_file, "rb") as image_file:
                     encoded_string = base64.b64encode(image_file.read())
              st.markdown(
              f"""
              <style>
              .stApp {{
                     background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
                     background-size: cover
              }}
              </style>
              """,
              unsafe_allow_html=True
              )


       add_bg_from_local('BG Image.jpg')  


# From here code for our project starts

model = pickle.load(open('model.pkl','rb'))
encoder = pickle.load(open('target_encoder.pkl','rb'))
transformer = pickle.load(open('transformer.pkl','rb'))


with col2:
      
       st.title("Bike Share Prediction")

       season = st.selectbox("Please Select the season",("spring","summer","fall","winter"))

       yr = st.selectbox("Select the year",(2011, 2012))

       mnth = st.selectbox("Select the month",('Jan','feb','mar','april', 'may','june','july','aug','sep','oct', 'nov', 'dec'))

       holiday = st.selectbox("Is there a holiday",('yes','no'))

       weekday = st.selectbox("Select Weekday", ('sunday', 'monday', 'tuesday', 'wednesday', 'thursday',
              'friday','saturday'))

       workingday = st.selectbox("Is Working day or not",('yes','no'))

       weathersit = st.selectbox("Select weathersit",(2,1,3))

       temp = st.text_input("Enter temperature between (0 - 0.5)",0.363)
       temp = float(temp)

       atemp = st.text_input("Enter atemperature between (0 - 0.5)",0.189)
       atemp = float(atemp)

       hum = st.text_input("Enter humadity between (0 - 0.8)",0.805)
       hum = float(hum)

       windspeed = st.text_input("Enter the windspeed (0 - 0.5)",0.304)
       windspeed = float(windspeed)

       casual = st.text_input("Enter casual number (0 - 500)",222)

       registered = st.text_input("Enter registered number (0-1600)",891)


# We have to store all in dictionary
l = {}
l['season'] = season
l['yr'] = yr
l['mnth'] = mnth
l['holiday'] = holiday
l['weekday'] = weekday
l['workingday'] = workingday
l['weathersit'] = weathersit
l['temp'] = temp
l['atemp'] = atemp
l['hum'] = hum
l['windspeed'] = windspeed
l['casual'] = casual
l['registered'] = registered


# store dictionary in dataframe
df = pd.DataFrame(l,index=[0])

# we do encoding of all features
# df['season'] = encoder.transform(df['season'])
# df['mnth'] = encoder.transform(df['mnth'])
df['holiday'] = df['holiday'].map({'yes':1,'no':0})
# df['weekday'] = encoder.transform(df['weekday'])
df['workingday'] = df['workingday'].map({'yes':1,'no':0})
df['season'] = df['season'].map({'spring':0, 'summer':1,'fall':2,'winter':3})
df['mnth'] = df['mnth'].map({'Jan':0,'feb':1,'mar':2,'april':3, 'may':4,'june':5,'july':6,'aug':7,'sep':8,'oct':9, 'nov':10, 'dec':11})
df['weekday'] = df['weekday'].map({'sunday':0, 'monday':1, 'tuesday':2, 'wednesday':3, 'thursday':4,'friday':5,'saturday':6})



df = transformer.transform(df)

y_pred = model.predict(df)

# Add Submit Button
with col2:
       if st.button("Show Counts"):
              st.header(f"{round(y_pred[0])} bikes will be shared")