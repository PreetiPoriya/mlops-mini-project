import dagshub
import mlflow

dagshub.init(repo_owner='PreetiPoriya', repo_name='mlops-mini-project', mlflow=True)


with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)