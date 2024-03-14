# test_score.py

import unittest
import joblib
from joblib import load
from score import score

import numpy as np
import os
import requests
import unittest
import time

class TestScore(unittest.TestCase):

    # load the pretrained model
    def setUp(self):
        self.model = joblib.load("best_model.pkl")
        self.text = "You've been selected for a special discount on luxury watches. Act now to save big!"
        self.threshold = 0.55
        self.pred, self.prop = score(self.text, self.model, self.threshold)

    # Smoke test to check if score function produces output without crashing
    def smoke_test(self):
        assert self.pred != None
        assert self.prop != None

    # Format test to ensure all input/output formats are as expected
    def format_test(self):
        assert type(self.text) == str 
        assert type(self.threshold) == float
        assert type(self.prop) == float
        assert self.pred == 0 or self.pred == 1

    # check if prediction values are 0 or 1
    def test_pred_values(self):
        assert self.pred == 0 or self.pred == 1

    # check if propensity score is between 0 and 1
    def test_prop_values(self):
        assert self.prop >= 0 and self.prop <= 1

    # check is threshold is 0, prediction becomes 1
    def test_threshold_0(self):
        pred, prop = score(self.text, self.model, 0)
        assert pred == 1

    # check is threshold is 1, prediction becomes 0
    def test_threshold_1(self):
        pred, prop = score(self.text, self.model, 1)
        assert pred == 0

    # testing obvious spam
    def test_spam(self):
        spam_text = "Congratulations! You have won a vacation to the Bahamas! Click here to win the prize!"
        pred, prop = score(spam_text, self.model, self.threshold)
        assert pred == 1

    # testing obvious non-spam
    def test_non_spam(self):
        non_spam_text = "Hi there! Just checking in to see how you're doing."
        pred, prop = score(non_spam_text, self.model, self.threshold)
        assert pred == 0



class TestFlaskIntegration(unittest.TestCase):
    def setUp(self):
        self.text = "You've been selected for a special discount on luxury watches. Act now to save big!"
        self.threshold = 0.55
        os.system('python app.py &')
        time.sleep(1)

    def tearDown(self):
        os.system("kill $(lsof -t -i:5000)")

    def test_flask_endpoint(self):
        response = requests.post('http://localhost:5000/score', json={'text': self.text, 'threshold': self.threshold})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('Prediction', data)
        self.assertIn('Propensity', data)

if __name__ == '__main__':
    unittest.main()