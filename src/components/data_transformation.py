import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from src.utils import save_object


# ----------------------------------------------------------
# CONFIG — Phase 3, Module 7
# Answers only WHERE the preprocessor artifact gets saved.
# os.path.join() ensures cross platform path — Module 4
# ----------------------------------------------------------
@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")


# ----------------------------------------------------------
# COMPONENT — Phase 3, Module 6
# Bundles all transformation logic into one self contained unit.
# Owns a config object — never hardcodes paths itself.
# ----------------------------------------------------------
class DataTransformation:

    def __init__(self):
        # Component owns its config — separates WHERE from HOW
        # Module 6 & 7
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer(self):
        """
        Builds and returns the preprocessing object.
        Separated from initiate_data_transformation() intentionally
        so the preprocessor logic can be tested and modified
        independently of the orchestration logic.
        """
        try:
            logging.info("Building preprocessing pipelines")

            # --------------------------------------------------
            # COLUMN DEFINITIONS
            # Nominal columns → OneHotEncoder
            # No natural ranking exists between categories.
            # Encoding as numbers would imply false hierarchy.
            #
            # Ordinal columns → OrdinalEncoder
            # A real world ranking exists between categories.
            # Numbers assigned must reflect that ranking.
            #
            # Numerical columns → StandardScaler
            # Already numbers but on different scales.
            # Scaling ensures no feature dominates due to magnitude.
            # --------------------------------------------------
            ohe_columns     = ["gender", "race_ethnicity", "lunch"]
            ordinal_columns = ["parental_level_of_education", "test_preparation_course"]
            scale_columns   = ["reading_score", "writing_score"]

            # --------------------------------------------------
            # ORDINAL RANKINGS
            # The order of values in each list defines the ranking.
            # OrdinalEncoder assigns 0, 1, 2... in this exact order.
            # Getting this wrong silently corrupts the model's learning.
            # --------------------------------------------------
            education_rank = [
                "some high school",
                "high school",
                "some college",
                "associate's degree",
                "bachelor's degree",
                "master's degree",
            ]
            prep_rank = ["none", "completed"]

            # --------------------------------------------------
            # NOMINAL PIPELINE — for ohe_columns
            # Step 1: SimpleImputer → fills missing values with
            #         the most frequent category in that column
            # Step 2: OneHotEncoder → creates one binary column
            #         per category, no ranking implied
            #
            # Pipeline ensures these steps always fire in order
            # and are always applied together — Module 13
            # --------------------------------------------------
            one_hot_pipeline = Pipeline(
                steps=[
                    ("imputer",         SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                ]
            )

            # --------------------------------------------------
            # ORDINAL PIPELINE — for ordinal_columns
            # Step 1: SimpleImputer → fills missing values with
            #         the most frequent category in that column
            # Step 2: OrdinalEncoder → assigns numbers that
            #         preserve the real world ranking we defined above
            # --------------------------------------------------
            ordinal_pipeline = Pipeline(
                steps=[
                    ("imputer",          SimpleImputer(strategy="most_frequent")),
                    ("ordinal_encoder",  OrdinalEncoder(
                        categories=[education_rank, prep_rank]
                    )),
                ]
            )

            # --------------------------------------------------
            # NUMERICAL PIPELINE — for scale_columns
            # Step 1: SimpleImputer → fills missing values with
            #         the median (robust to outliers unlike mean)
            # Step 2: StandardScaler → transforms values to have
            #         mean=0 and std=1 so no feature dominates
            #         the model due to scale differences
            # --------------------------------------------------
            scaler_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler",  StandardScaler()),
                ]
            )

            # --------------------------------------------------
            # COLUMN TRANSFORMER
            # Applies each pipeline to its designated columns
            # simultaneously and combines outputs into one array.
            # This is the single object we save as preprocessor.pkl
            # — it carries all three pipelines inside it.
            # --------------------------------------------------
            preprocessor = ColumnTransformer(
                [
                    ("one_hot_pipeline", one_hot_pipeline, ohe_columns),
                    ("ordinal_pipeline", ordinal_pipeline, ordinal_columns),
                    ("scaler_pipeline",  scaler_pipeline,  scale_columns),
                ]
            )

            logging.info("Preprocessing pipelines built successfully")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys) # type: ignore

    def initiate_data_transformation(self, train_path, test_path):
        """
        Orchestrates the full transformation flow:
        reads artifacts → builds preprocessor → fits and transforms
        → concatenates target → saves preprocessor → returns arrays
        """
        try:
            # Read the artifacts produced by DataIngestion — Module 11
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)
            logging.info("Read train and test data")

            # Get the preprocessor object — defined separately
            # so this method stays focused on orchestration only
            preprocessor = self.get_data_transformer()
            logging.info("Preprocessing object obtained")

            # --------------------------------------------------
            # FEATURE / TARGET SEPARATION
            # X → input features the model learns from
            # y → target the model is trying to predict
            # We transform X only — never touch y
            # --------------------------------------------------
            target_column = "math_score"

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            X_test  = test_df.drop(columns=[target_column])
            y_test  = test_df[target_column]

            logging.info(f"Training input data shape: {X_train.shape}")
            logging.info(f"Testing input data shape: {X_test.shape}")

            # --------------------------------------------------
            # FIT ON TRAIN, TRANSFORM BOTH
            # fit_transform on train → learns statistics (mean,
            # std, categories) FROM training data only, then applies them
            #
            # transform on test → applies the SAME statistics
            # learned from train — never learns from test data
            #
            # Fitting on test data would be data leakage —
            # the model would have seen test information during
            # training, making evaluation results unreliable
            # --------------------------------------------------
            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr  = preprocessor.transform(X_test)
            logging.info("Input training and testing features transformed")

            # --------------------------------------------------
            # CONCATENATE FEATURES AND TARGET
            # np.c_[] joins arrays column-wise
            # Result: [transformed_features | target]
            # This single array gets passed to ModelTrainer
            # --------------------------------------------------
            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr  = np.c_[X_test_arr,  np.array(y_test)]

            # --------------------------------------------------
            # SAVE PREPROCESSOR ARTIFACT — Module 12
            # save_object() from utils.py uses dill to serialize
            # the fitted ColumnTransformer to disk.
            # At prediction time, load_object() thaws it back
            # and applies the exact same transformations to
            # live user input — ensuring training/serving consistency
            # --------------------------------------------------
            preprocessor_file_path = self.transformation_config.preprocessor_path
            save_object(preprocessor_file_path, preprocessor)
            logging.info(f"Preprocessor object saved at: {preprocessor_file_path}")

            # Return transformed arrays for ModelTrainer
            # and the path so the pipeline knows where preprocessor lives
            return train_arr, test_arr, self.transformation_config.preprocessor_path

        except Exception as e:
            raise CustomException(e, sys) # type: ignore