# CD4ML-AWS-Serverless
CD4ML Scenario for Serverless Machine Learning on AWS.    
The Jupyter notebooks must run in AWS SageMaker.     

## PrepareData.ipnb
Download raw data from a public AWS S3 bucket. Split the data into train and test data, encode non-numerical values and safe the data on a personal AWS S3 bucket.

## TrainModel.ipynb
Train and evaluate different Machine Learning models in the Jupyter notebook.

## TrainModelSM.ipynb
Run the training on an extra instance in a docker container.

## TrainModelSMExp.ipynb
Run the training on an extra instance in a docker container and monitor the experiments and trials in SageMaker Studio.

## Setting up the Web Application

### On your local laptop:
```
cd CD4ML-AWS-Serverless
```
start virtenv flask:
```
source flask/bin/activate
```
Set environment variables and run the flask server:
```
export FLASK_APP=flaskApp/app.py
export FLASK_ENV=development
flask run
```
At the end, deactivage the virtualenv:
````
deactivate
````
