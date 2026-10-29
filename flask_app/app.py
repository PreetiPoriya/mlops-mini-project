from flask import Flask, render_template, request
import mlflow
import dagshub
import pickle
from preprocessing_utility import normalize_text
import os

# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "PreetiPoriya"
repo_name = "mlops-mini-project"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

app = Flask(__name__)


# load model from model registry using the "staging" alias
model_name = "my_model"
model = mlflow.pyfunc.load_model(f"models:/{model_name}@production")

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