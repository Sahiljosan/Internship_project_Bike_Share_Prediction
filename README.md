# Bike_Share_Prediction
Intership Project <br>
Deployment URL
```
https://sahiljosan-internship-project-bike-share-prediction-app-oa6k99.streamlit.app/
```
## UI of Project
[![](https://i.imgur.com/jwEZJCi.jpg])](https://youtu.be/WswH6Z0Z1-o)

## Table of content
- [About Project](#about-project)
- [Documentation](#documentation)
- [Software and Account Requirement](#software-and-account-requirement)
- [Tools used](#tools-used)
- [Project Architecture](#project-architecture) 
- [Project Pipeline](#project-pipeline)


## About Project
Bike sharing systems are new generation of traditional bike rentals where whole process from membership, rental and return back has become automatic. Through these systems, user is able to easily rent a bike from a particular position and return back at another position. <br>

Apart from interesting real world applications of bike sharing systems, the characteristics of data being generated by these systems make them attractive for the research. Opposed to other transport services such as bus or subway, the duration of travel, departure and arrival position is explicitly recorded in these systems. This feature turns bike sharing system into a virtual sensor network that can be used for sensing mobility in the city. <br>

I am considering variables as season, yr,mnth, holiday, weekday, workingday, weathersit, Temp, atemp, hum, windspeed, casual, registered to predict the count of total rental bikes That will be required. <br>

[`Back to top`](#table-of-content)

## Documentation
- [Architechture Design](https://drive.google.com/file/d/1fjl6Bh-yNueGRfbugD6XAvTm1qynajng/view?usp=sharing)
- [Detailed Project Report](https://drive.google.com/file/d/1eqQ_QtK96_5nM0ilYaSKrAS-N2-gmprO/view?usp=sharing)
- [High-Level Design (HLD)](https://drive.google.com/file/d/1BatDRodWA36FKkqk4Vdf-n4leqmFdY1L/view?usp=sharing)
- [Low-Level Design (LLD)](https://drive.google.com/file/d/1f1HejFCB5Rxy3mlcs5Akijsl8Y4Nfzih/view?usp=sharing)
- [Wireframe](https://drive.google.com/file/d/1MycaOO3XoNX3VWBA9PockZFZXH6s35Z3/view?usp=sharing)

[BACK TO TOP](#table-of-content)

## Software and Account Requirement 
1. [Github Account](https://github.com/Sahiljosan)
2. [Streamlit Account](https://streamlit.io/)
3. [VS Code IDE](https://code.visualstudio.com/download)
4. [GIT CLI](https://git-scm.com/downloads)

[BACK TO TOP](#table-of-content)

## Tools used
Python programming language and frameworks such as Numpy, Pandas, Scikit-learn, VS Code, git and streamlit are used to build the whole model <br><br>
![](https://i.imgur.com/EutJMW9.jpg)
- VS Code is used as IDE.
- For visualization of the plots, Matplotlib, Seaborn and Plotly are used.
- Streamlit is used for deployment of the model.
- Front end development is done using Streamlit.
- Python Flask is used for backend development.
- GitHub is used as version control system.

[BACK TO TOP](#table-of-content)

## Project Architecture 
![](https://i.imgur.com/t3jmkkO.png)
`Data Colleciton` <br>
The data for this project is collected from UC Irvine Machine Learning Repository. The link for the data is https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

`Data Description` <br>
Bike sharing dataset is publicaly available on UC Irvine Machine Learning Repository. This dataset contains the hourly and daily count of rental bikes between years 2011 and 2012 in Capital bikeshare system with the corresponding weather and seasonal information.

`Data Pre-processing` <br>
    - Checked the datatype of features in dataset using df.info()
    - Checked for Null values, because the null values can affect the accuracy of the model.
    - Perform Label Encoding for the features that have categorical data.
    - Checked the distribution of the features to interpret its importance.
    Now, the data is prepared to train a machine learning model.
    
`Modelling Process` <br>
After pre-processing the data, we visualize our data to gain insights and split into two parts, train and test data. After Splitting the data, we use different machine learning algorithms like – Linear Regression, Random Forest Regressor, Decision Tree Regressor to predict the Count of Bike share <br><br>
![](https://i.imgur.com/RZK4Jxl.png)

[BACK TO TOP](#table-of-content)

## Project Pipeline 
1. [Data Ingestion](#data-ingestion)
2. [Data Validation](#data-validation)
3. [Data Transformation](#data-transformation)
4. [Model Training](#model-training)
5. [Model Evaluation](#model-evaluation)
6. [Model Deployment](#model-deployment)

### Data Ingestion
In the Ingestion Process, we will convert our original dataset which is in Zip format to csv format. After that we will split them into train and test dataset.

### Data Validation
In Data validation steps we could use Null value handling, outlier handling, Imbalanced data set handling, Handling columns with standard deviation zero or below a threshold, etc.

### Data Transformation
In this step we will transform out data. We will use standard scaler for numeric data and we will convert categorical data into numeric data using label encoding technique so that machine can understand it.

### Model Training
Here we will build the Machine Learning model using all regression algorithms.<br>
![](https://i.imgur.com/u11UVgW.png)

[BACK TO TOP](#table-of-content)

### Model Evaluation
Here model evaluation will be done on the model which we got in the model building stage. We can define base accuracy of the model and if model accuracy is higher then base accuracy, then only our model will accept otherwise it will be rejected.

### Model Deployment
Here model will be deployed to Streamlit cloud platform.<br><br>
![](https://i.imgur.com/AZ8RcV7.png)
<br>
Thanks and Regards <br>
`Sahil Josan`

[BACK TO TOP](#table-of-content)



