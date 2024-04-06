# Save Model Using joblib
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import SGDClassifier
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer


df =  pd.read_csv('emails.csv')
X = df['text'].copy()
y = df['spam'].copy()

def preprocessing(text):
    new_text = ''
    for char in text[8:]:
        if char.isalnum() or char == " ":
            new_text += char
    return new_text

custom_preprocessing_transformer = FunctionTransformer(preprocessing)

preprocessor = ColumnTransformer(transformers=[
    ('preprocess', custom_preprocessing_transformer, [0]),  
    ('tfidf', TfidfVectorizer(), [0])
    ],
    remainder='passthrough'  
)

# Define the final pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),  
    ('sgd', SGDClassifier(loss='log_loss', random_state=42))  
])

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state=2024)

pipeline.fit(train_X, train_y)

joblib.dump(pipeline, 'best_model.pkl')