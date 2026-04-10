import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


# ----------------------------------------------------------
# CONFIG — Phase 3, Module 7
# Answers only WHERE the model artifact gets saved.
# os.path.join() ensures cross platform path — Module 4
# ----------------------------------------------------------
@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artifacts", "model.pkl")


# ----------------------------------------------------------
# MODELS AND PARAMETERS — Phase 3, Module 6
# A dedicated class that owns the models and their
# hyperparameter grids. Separates WHAT to try from
# HOW to train — keeping ModelTrainer clean and focused.
#
# Using __init__ ensures each instance gets its own
# fresh copy of models and parameters — avoids subtle
# bugs from shared class-level state.
# ----------------------------------------------------------
class ModelsAndParameters:
    def __init__(self):
        self.models = {
            "Random Forest":          RandomForestRegressor(),
            "Decision Tree":          DecisionTreeRegressor(),
            "Gradient Boosting":      GradientBoostingRegressor(),
            "Linear Regression":      LinearRegression(),
            "XGBRegressor":           XGBRegressor(),
            "CatBoosting Regressor":  CatBoostRegressor(verbose=False),
            "AdaBoost Regressor":     AdaBoostRegressor(),
        }

        self.parameters = {
            "Decision Tree": {
                "criterion":    ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                "splitter":     ["best", "random"],
                "max_features": ["sqrt", "log2"],
            },
            "Random Forest": {
                "criterion":    ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                "max_features": ["sqrt", "log2", None],
                "n_estimators": [8, 16, 32, 64, 128, 256],
            },
            "Gradient Boosting": {
                "loss":          ["squared_error", "huber", "absolute_error", "quantile"],
                "learning_rate": [0.1, 0.01, 0.05, 0.001],
                "subsample":     [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                "criterion":     ["squared_error", "friedman_mse"],
                "max_features":  ["sqrt", "log2"],
                "n_estimators":  [8, 16, 32, 64, 128, 256],
            },
            "Linear Regression": {},  # no hyperparameters to tune
            "XGBRegressor": {
                "learning_rate": [0.1, 0.01, 0.05, 0.001],
                "n_estimators":  [8, 16, 32, 64, 128, 256],
            },
            "CatBoosting Regressor": {
                "depth":         [6, 8, 10],
                "learning_rate": [0.01, 0.05, 0.1],
                "iterations":    [30, 50, 100],
            },
            "AdaBoost Regressor": {
                "learning_rate": [0.1, 0.01, 0.5, 0.001],
                "loss":          ["linear", "square", "exponential"],
                "n_estimators":  [8, 16, 32, 64, 128, 256],
            },
        }


# ----------------------------------------------------------
# COMPONENT — Phase 3, Module 6
# Owns model training orchestration only.
# Delegates model definitions to ModelsAndParameters
# and shared operations to utils.py — Module 8
# ----------------------------------------------------------
class ModelTrainer:

    def __init__(self):
        # Config tells this component WHERE to save the model
        self.model_trainer_config = ModelTrainerConfig()

        # Separate class owns WHAT models and params to try
        self.models_params = ModelsAndParameters()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            # --------------------------------------------------
            # ARRAY SPLITTING
            # DataTransformation returned arrays shaped as:
            # [transformed_features | target]
            # [:, :-1]  → all rows, all columns except last = X
            # [:, -1]   → all rows, last column only = y
            # --------------------------------------------------
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )
            logging.info("Input and target features separated for training and testing")

            models = self.models_params.models
            params = self.models_params.parameters

            # --------------------------------------------------
            # MODEL EVALUATION — Module 8, utils.py
            # evaluate_models() trains every model with
            # GridSearchCV, scores on test set, returns:
            # score_report  → {model_name: r2_score}
            # params_report → {model_name: best_params}
            # --------------------------------------------------
            score_report, params_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params,
            )
            logging.info("Model and parameters evaluation report generated")

            # --------------------------------------------------
            # BEST MODEL SELECTION
            # max() with key finds the model name with the
            # highest score in one clean step — no list
            # comprehension needed, result is a plain string
            # --------------------------------------------------
            best_model_name  = max(score_report, key=lambda name: score_report[name])
            best_model_score = score_report[best_model_name]
            best_model       = models[best_model_name]
            best_params      = params_report[best_model_name]

            # --------------------------------------------------
            # VALIDATION GATE
            # Reject the pipeline if no model meets the minimum
            # quality threshold — prevents saving a bad model.
            # Check happens BEFORE logging success or saving.
            # --------------------------------------------------
            logging.info(
                f"Best model: {best_model_name} with R2 score: {best_model_score:.2f}"
            )
            if best_model_score < 0.6:
                raise CustomException("No best model found", sys) # type: ignore

            logging.info("Best model validation passed, proceeding to save")

            # --------------------------------------------------
            # REFIT BEST MODEL ON FULL TRAINING DATA
            # set_params() applies the winning hyperparameters
            # fit() trains the model — must happen after
            # set_params() otherwise predict() will crash
            # on an unfitted model
            # --------------------------------------------------
            best_model.set_params(**best_params)
            best_model.fit(X_train, y_train)

            # --------------------------------------------------
            # SAVE MODEL ARTIFACT — Module 12
            # save_object() uses dill to serialize the fitted
            # model to disk. load_object() in prediction
            # pipeline thaws it back for inference.
            # --------------------------------------------------
            save_object(self.model_trainer_config.model_path, best_model)
            logging.info(f"Model saved at: {self.model_trainer_config.model_path}")

            # --------------------------------------------------
            # FINAL EVALUATION
            # One last score on the test set using the refitted
            # best model — this is the number we report back
            # to the training pipeline as the final metric
            # --------------------------------------------------
            predicted  = best_model.predict(X_test)
            r2_square  = r2_score(y_true=y_test, y_pred=predicted)
            logging.info(f"Final R2 score on test set: {r2_square:.4f}")

            return r2_square

        except Exception as e:
            raise CustomException(e, sys) # type: ignore