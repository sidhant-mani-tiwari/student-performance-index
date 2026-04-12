import sys
import os
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


# ----------------------------------------------------------
# CONFIG — Phase 3, Module 7
# Answers only WHERE the model and preprocessor artifacts
# live on disk. Separates "where" from "how" — keeps
# predict() free of hardcoded paths.
# os.path.join() ensures cross platform paths — Module 4
# @dataclass auto generates __init__ so field annotations
# become real instance attributes with default values
# ----------------------------------------------------------
@dataclass
class PredictPipelineConfig:
    model_path:        str = os.path.join("artifacts", "model.pkl")
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")


# ----------------------------------------------------------
# PREDICTION PIPELINE — Phase 5
# The inference engine. Runs at prediction time, not
# training time. Mirrors the training pipeline but in
# reverse — loads frozen artifacts and applies them
# to live user input to generate a prediction.
#
# Flow:
# raw user input (dataframe)
#       ↓ preprocessor.transform()
# scaled/encoded array
#       ↓ model.predict()
# prediction
# ----------------------------------------------------------
class PredictPipeline:

    def __init__(self) -> None:
        # Own the config — paths never hardcoded in logic
        # Module 6 & 7
        self.pipeline_config = PredictPipelineConfig()

    def predict(self, features):
        try:
            # Retrieve artifact paths from config
            # never hardcoded here — Module 7
            model_path        = self.pipeline_config.model_path
            preprocessor_path = self.pipeline_config.preprocessor_path

            logging.info("Loading model and preprocessor artifacts")

            # --------------------------------------------------
            # LOAD ARTIFACTS — Module 12
            # load_object() uses dill to deserialize both
            # artifacts from disk back into memory.
            # These are the exact objects saved during training
            # — the preprocessor carries all fitted statistics
            # (means, std, categories) learned from training data
            # --------------------------------------------------
            model        = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            logging.info("Model and preprocessor loaded successfully")

            # --------------------------------------------------
            # TRANSFORM USER INPUT — Module 13
            # preprocessor.transform() applies the SAME
            # transformations used during training:
            # - OHE on nominal columns
            # - Ordinal encoding on ordinal columns
            # - Scaling on numerical columns
            #
            # This is why we saved the preprocessor — to
            # guarantee training and serving consistency.
            # We use transform() NOT fit_transform() here
            # because the preprocessor is already fitted
            # on training data. Fitting again on user input
            # would be data leakage.
            # --------------------------------------------------
            transformed_data = preprocessor.transform(features)

            # --------------------------------------------------
            # GENERATE PREDICTION
            # model.predict() returns a numpy array of
            # predictions — one value per row of input
            # --------------------------------------------------
            prediction = model.predict(transformed_data)
            logging.info(f"Prediction generated successfully: {prediction}")

            return prediction

        except Exception as e:
            raise CustomException(e, sys) # type: ignore


# ----------------------------------------------------------
# CUSTOM DATA — Phase 5
# The bridge between raw user input and a properly
# structured dataframe that the preprocessor expects.
#
# Two responsibilities:
# 1. Accept and store raw user input values in __init__
# 2. Package them into a correctly named dataframe
#    via get_data_as_dataframe()
#
# Column names in the dictionary must match EXACTLY
# what the preprocessor was trained on — any mismatch
# will cause a silent failure or crash at transform time.
# ----------------------------------------------------------
class CustomData:

    def __init__(
        self,
        gender:                      str,
        race_ethnicity:              str,
        parental_level_of_education: str,
        lunch:                       str,
        test_preparation_course:     str,
        reading_score:               float,
        writing_score:               float,
    ):
        # Store each field as an instance attribute
        # Type annotations document exactly what the
        # web form must provide — str vs int matters
        # because the preprocessor expects specific types
        self.gender                      = gender
        self.race_ethnicity              = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch                       = lunch
        self.test_preparation_course     = test_preparation_course
        self.reading_score               = reading_score
        self.writing_score               = writing_score

    def get_data_as_dataframe(self):
        try:
            # --------------------------------------------------
            # BUILD INPUT DICTIONARY
            # Each key must match the exact column name the
            # preprocessor saw during training — Module 13
            # Values wrapped in lists because pd.DataFrame
            # expects sequences, not scalar values.
            # Result is a single row dataframe representing
            # one user's prediction request.
            # --------------------------------------------------
            custom_data_input_dict = {
                "gender":                      [self.gender],
                "race_ethnicity":              [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch":                       [self.lunch],
                "test_preparation_course":     [self.test_preparation_course],
                "reading_score":               [self.reading_score],
                "writing_score":               [self.writing_score],
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info("User input converted into dataframe successfully")
            return df

        except Exception as e:
            raise CustomException(e, sys) # type: ignore