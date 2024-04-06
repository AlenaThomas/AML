import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from typing import Tuple

def preprocessing(text):
    new_text = ''
    for char in text:
        if char.isalnum() or char == " ":
            new_text += char
    return new_text

def score(text: str, model: BaseEstimator, threshold: float) -> Tuple[bool, float]:
    '''
    Scores a trained model on a given text

    Arguments:
    - text (str) - the input text to be scored
    - model (sklearn.estimator) - the trained model
    - threshold (float) - decision threshold for prediction

    Returns:
    - (bool, float) - a tuple containing the prediction (0 or 1) and the propensity score
    '''

    X = pd.DataFrame(np.array([text]).reshape(-1,1))

    X_new = X.iloc[:, 0].apply(preprocessing)

    propensity = model.predict_proba(X_new)[0, 1]

    if propensity >= threshold:
        prediction = 1
    else:
        prediction = 0

    return (prediction, propensity)