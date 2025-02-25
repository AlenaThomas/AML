from flask import Flask, request
import joblib
from score import score
import json

model = joblib.load("best_model.pkl")

app = Flask(__name__)

@app.route('/score', methods=['POST'])
def score_endpoint():
    data = json.loads(request.data)
    text = data['text']
    threshold = data['threshold']
    pred, prop = score(text, model, threshold)
    return {'Prediction': int(pred), 'Propensity': float(prop)}

if __name__ == '__main__':
    app.run(debug=True)