# CD4ML-AWS-Serverless
CD4ML Scenario for Serverless Machine Learning on AWS.    
The Jupyter notebooks must run in AWS SageMaker.     

## PrepareData.ipnb
Download raw data from a public AWS S3 bucket. Split the data into train and test data, encode non-numerical values and safe the data on a personal AWS S3 bucket.

## TrainModel.ipynb
Train and evaluate different Machine Learning models in the Jupyter notebook.

## TrainModelSM.ipynb
Run the training on an extra instance in a docker container.
