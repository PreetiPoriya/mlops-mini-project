# register model (adds the "staging" alias to the version registered during evaluation)

import json
import logging
import mlflow
import dagshub

dagshub.init(repo_owner='PreetiPoriya', repo_name='mlops-mini-project', mlflow=True)

# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def set_staging_alias(model_name: str, model_info: dict):
    """Add the 'staging' alias to the version created by this run."""
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")

        # Prefer the version created by this run, otherwise take the latest
        match = [v for v in versions if v.run_id == model_info['run_id']]
        version = match[0] if match else max(versions, key=lambda v: int(v.version))

        client.set_registered_model_alias(model_name, "staging", version.version)
        logger.debug('Model %s version %s aliased as staging.', model_name, version.version)
    except Exception as e:
        logger.error('Error while setting the staging alias: %s', e)
        raise

def main():
    try:
        model_info = load_model_info('reports/experiment_info.json')
        set_staging_alias("my_model", model_info)
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        raise  # makes the DVC stage fail visibly instead of reporting success

if __name__ == '__main__':
    main()