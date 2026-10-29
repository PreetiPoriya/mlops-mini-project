from flask import Flask, render_template, request
import mlflow
import pickle
import os
 
# Works both when run from the repo root (tests / CI) and from inside flask_app/
try:
    from flask_app.preprocessing_utility import normalize_text
except ModuleNotFoundError:
    from preprocessing_utility import normalize_text
 
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")
 
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
 
dagshub_url = "https://dagshub.com"
repo_owner = "PreetiPoriya"
repo_name = "mlops-mini-project"
 
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
 
app = Flask(__name__)
 
# load model from the model registry using the "production" alias
model_name = "my_model"
model = mlflow.pyfunc.load_model(f"models:/{model_name}@production")
 
# load the vectorizer saved during feature engineering (path independent of cwd)
with open(os.path.join(BASE_DIR, "models", "vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)
 
 
@app.route("/")
def home():
    return render_template("index.html", result=None)
 
 
@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["text"]
 
    # clean
    text = normalize_text(text)
 
    # bag of words
    features = vectorizer.transform([text])
 
    # prediction
    result = model.predict(features)
    prediction = result[0]
 
    # works whether the model returns text labels or 1/0
    is_happy = 1 if prediction in ("happiness", 1) else 0
 
    return render_template("index.html", result=is_happy)
 
 
if __name__ == "__main__":
    app.run(debug=True)