from flask import Flask, render_template, request
import mlflow
import dagshub
import pickle
from preprocessing_utility import normalize_text

dagshub.init(repo_owner='PreetiPoriya', repo_name='mlops-mini-project', mlflow=True)

app = Flask(__name__)

# load model from model registry using the "staging" alias
model_name = "my_model"
model = mlflow.pyfunc.load_model(f"models:/{model_name}@staging")

# load the vectorizer saved during feature engineering
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html', result=None)


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']

    # clean
    text = normalize_text(text)

    # bow
    features = vectorizer.transform([text])

    # prediction
    result = model.predict(features)
    prediction = result[0]
    print("Raw prediction:", prediction)  # you can remove this later

    # works whether the model returns text labels or 1/0
    is_happy = 1 if prediction in ('happiness', 1) else 0

    # show
    return render_template('index.html', result=is_happy)


if __name__ == '__main__':
    app.run(debug=True)