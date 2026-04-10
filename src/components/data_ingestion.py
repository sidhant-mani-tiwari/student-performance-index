# ============================================================
# IMPORTS
# os & pathlib  → Phase 2, Module 4: File system navigation
# sys           → Phase 2, Module 5: Feeds crash location into CustomException
# CustomException → Phase 4, Module 10: Our custom error detective
# logging       → Phase 4, Module 9: Our persistent black box diary
# dataclass     → Phase 3, Module 7: Separates "where" (config) from "how" (logic)
# ============================================================
import os
import sys
from pathlib import Path
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


# ============================================================
# CONFIG CLASS — Phase 3, Module 7: The Config Pattern
# @dataclass automatically generates __init__, __repr__ etc.
# so we don't have to write boilerplate ourselves.
#
# This class answers only ONE question: "WHERE do files live?"
# It has zero logic — just path declarations.
#
# os.path.join() used here (not pathlib) because dataclass
# fields need simple default values — plain strings work best.
# This is cross-platform safe (no hardcoded / or \)
# ============================================================
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path:  str = os.path.join('artifacts', 'test.csv')
    raw_data_path:   str = os.path.join('artifacts', 'raw.csv')


# ============================================================
# MAIN CLASS — Phase 3, Module 6: OOP
# DataIngestion is a blueprint (class) that bundles:
#   - its configuration data  (self.ingestion_config)
#   - its logic               (initiate_data_ingestion)
# into one clean, self-contained unit.
# ============================================================
class DataIngestion:

    def __init__(self):
        # Instantiating the config dataclass gives this object
        # its own personal copy of all three file paths.
        # "self" = this object referring to its own data (Module 6)
        # This separates "where" (config) from "how" (logic below)
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        # INFO log → Phase 4, Module 9
        # Acts like a timestamped diary entry: "pipeline entered this step"
        logging.info("Entered the data ingestion component")

        try:
            # --------------------------------------------------
            # PATH RESOLUTION — Phase 2, Module 4
            # Path(__file__) → absolute path of THIS script
            # .parent.parent → navigate up two folders to project root
            # / operator     → pathlib's cross-platform path joiner
            #                  (not a string slash — it's a Python operator)
            # --------------------------------------------------
            basedir  = Path(__file__).parent.parent.parent
            datapath = basedir / "notebook" / "data" / "stud.csv"

            df = pd.read_csv(datapath)
            logging.info(f"Dataset read succesfully -- Data shape: {df.shape}")

            # --------------------------------------------------
            # DIRECTORY CREATION — Phase 2, Module 4
            # os.path.dirname() strips filename → gives just the folder
            # 'artifacts/train.csv' → 'artifacts'
            #
            # os.makedirs() creates the folder and ALL missing
            # parent folders in one shot.
            # exist_ok=True → don't crash if folder already exists
            #
            # We create the DIRECTORY here, not the file.
            # The file gets created below when we call .to_csv()
            # --------------------------------------------------
            artifact_dir = os.path.dirname(self.ingestion_config.train_data_path)
            os.makedirs(artifact_dir, exist_ok=True)

            # Save a raw unmodified copy of the data as a checkpoint
            # self.ingestion_config.raw_data_path → 'artifacts/raw.csv'
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated.")

            # Split data → 80% train, 20% test, reproducible via random_state
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info(f"Train shape: {train_set.shape} | Test shape: {test_set.shape}")
            # Save both splits into artifacts/ as checkpoints (Module 11)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,  index=False, header=True)

            logging.info("Ingestion of data is completed.")

            # Return paths (not dataframes) so the next pipeline
            # step (DataTransformation) knows where to read from
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            # --------------------------------------------------
            # CUSTOM EXCEPTION — Phase 4, Module 10
            # We pass both the original error (e) AND the sys module.
            # sys gives CustomException access to sys.exc_info()
            # which extracts the exact FILE and LINE NUMBER of the crash
            # instead of a generic unhelpful error message.
            # --------------------------------------------------
            raise CustomException(e, sys) # type: ignore

if __name__ == "__main__":
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()
    print(f"Train: {train_path}")
    print(f"Test: {test_path}")