import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


class TrainingPipeline:
    def run_pipeline(self):
        try:
            print("Inside training pipeline")
            # Step 1: Data Ingestion
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()
            logging.info(f"Training path: {train_path} | Testing path: {test_path}")
            # Step 2: Data Transformation
            transformation = DataTransformation()
            logging.info("Train/test paths have been passed for transformation.")
            train_arr, test_arr, _ = transformation.initiate_data_transformation(train_path=train_path, test_path=test_path)
            logging.info("Data Transformed")
            # Step 3: Model Training
            trainer = ModelTrainer()
            logging.info("Model training started")
            r2_score = trainer.initiate_model_trainer(train_arr=train_arr, test_arr=test_arr)
            logging.info(f"Model training completed with R2 score of: {r2_score}")
            # Return final score
            return r2_score
        except Exception as e:
            raise CustomException(e, sys) # type: ignore 

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    r2_score = pipeline.run_pipeline()
    print(f"Training complete. Final R2 Score: {r2_score:.4f}")