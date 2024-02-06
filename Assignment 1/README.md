# Assignment 1: Build a prototype for email spam classification 

Link to dataset: https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset

In prepare.ipynb write the functions to 
1. load the data from a given file path
2. preprocess the data (if needed)
3. split the data into train/validation/test
4. store the splits at train.csv/validation.csv/test.csv

In train.ipynb write the functions to
1. fit a model on train data
2. score a model on given data
3. evaluate the model predictions
4. validate the model
   - fit on train
   - score on train and validation
   - evaluate on train and validation
   - fine-tune using train and validation (if necessary)
5. score 3 benchmark models on test data and select the best one
