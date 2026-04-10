#src/utils.py
import os
import sys
import dill
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def save_object(filepath: str, obj) -> None:
    """
    Serializes any python object to disk using dill
    dill extend pickle by handling lambda functions
    
    Called by:
        - DataTransformation  → saves preprocessor.pkl
        - ModelTrainer        → saves model.pkl
    
    file_path : destination path e.g. 'artifacts/model.pkl'
    obj       : any picklable Python object
    """
    try:
        # Ensure destination directory exists before writing
        # dirname() strips filename → gives just the folder
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)
        
        # Serialize the objects to bytes and write to disk
        with open(filepath, "wb") as f:
            dill.dump(obj, f)
        logging.info(f"Object saved successfully at: {filepath}")
    except Exception as e:
        raise CustomException(e, sys) # type: ignore

def load_object(filepath: str):
    """
    Loads a dilled object from disk back into memory.

    Called by:
        - PredictionPipeline → loads model.pkl + preprocessor.pkl

    file_path : path to the .pkl file
    returns   : the original Python object, fully reconstructed
    """
    try:
        # Verify the file actually exists before loading
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No file found at: {filepath}")
        # Read byte from disk and construct python object
        with open(filepath, "rb") as f:
            obj = dill.load(f)
        logging.info(f"Object successfully loaded from: {filepath}")
        return obj
    except Exception as e:
        raise CustomException(e, sys) # type: ignore
    
def evaluate_models(X_train, y_train, X_test, y_test, models: dict, params: dict):
    """
    Returns a report of models and their performance
    based on their R square score in the form of dictionary
    
    :param X_train: Input features from training set
    :param y_train: Target feature from training set
    :param X_test: Input features from testing set
    :param y_test: Target feature from testing set
    :param models: A dictionary of model_name: models
    :type models: dict
    :param params: A dictionary of model_name: param(Hyperparameters for tuning the model)
    :type params: dict
    """
    try:
        score_report = {}
        params_report = {}
        for model_name, model in models.items():
            param = params[model_name]
            
            logging.info(f"Training {model_name} with GridSearchCV")
            grid_search = GridSearchCV(estimator=model, param_grid=param, cv=3)
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_model.fit(X_train, y_train)
            
            y_test_pred = best_model.predict(X_test)
            
            test_model_score = r2_score(y_test, y_test_pred)
            logging.info(f"{model_name} → R2 Score: {test_model_score:.4f}")
            
            score_report[model_name] = test_model_score
            params_report[model_name] = best_params
            logging.info("Model evaluation completed")
        return score_report, params_report
    except Exception as e:
        raise CustomException(e, sys) # type: ignore    