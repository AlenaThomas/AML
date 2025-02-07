# Assignment 2: Experiment Tracking 

## DATA VERSION CONTROL

In prepare.ipynb track the versions of data using dvc
1. load the raw data into raw_data.csv and save the split data into train.csv/validation.csv/test.csv
2. update train/validation/test split by choosing different random seed
3. checkout the first version (before update) using dvc and print the distribution of target variable (number of 0s and number of 1s) in train.csv, validation.csv, and test.csv
4. checkout the updated version using dvc and print the distribution of target variable in train.csv, validation.csv, test.csv
5. bonus: (decouple compute and storage) track the data versions using google drive as storage

## MODEL VERSION CONTROL & EXPERIMENT TRACKING

in train.ipynb track the experiments and model versions using mlflow
1. build, track, and register 3 benchmark models using MLflow
2. checkout and print AUCPR for each of the three benchmark models

link to dataset: https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset

References
1. https://dvc.org/doc/start/data-management/data-versioning
2. https://realpython.com/python-data-version-control/	(important)
3. https://towardsdatascience.com/how-to-manage-files-in-google-drive-with-python-d26471d91ecd

1. https://mlflow.org/docs/latest/tracking.html
2. https://mlflow.org/docs/latest/getting-started/intro-quickstart/index.html
3. https://www.datarevenue.com/en-blog/how-we-track-machine-learning-experiments-with-mlflow
4. https://towardsdatascience.com/experiment-tracking-with-mlflow-in-10-minutes-f7c2128b8f2c
