# promote model

import os
import mlflow

def promote_model():
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

    client = mlflow.MlflowClient()
    model_name = "my_model"

    # Get the version currently aliased as staging
    staging_version = client.get_model_version_by_alias(model_name, "staging").version

    # Keep a record of the current production version (if any) as "archived"
    try:
        old_prod = client.get_model_version_by_alias(model_name, "production")
        client.set_registered_model_alias(model_name, "archived", old_prod.version)
        print(f"Previous production version {old_prod.version} marked as archived")
    except mlflow.exceptions.MlflowException:
        print("No existing production version found")

    # Promote: moving the alias automatically removes it from the old version
    client.set_registered_model_alias(model_name, "production", staging_version)
    print(f"Model version {staging_version} promoted to production")

if __name__ == "__main__":
    promote_model()