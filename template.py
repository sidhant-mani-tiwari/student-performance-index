import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [
    ".github/workflows/.gitkeep",
    "src/__init__.py",
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_transformation.py",
    "src/components/model_trainer.py",
    "src/pipelines/__init__.py",
    "src/pipelines/training_pipeline.py",
    "src/pipelines/prediction_pipeline.py",
    "src/logger.py",
    "src/exception.py",
    "src/utils.py",
    "app.py",
    "requirements.py",
    "setup.py"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath) # separates directory and filename from filepath
    
    # Logic to create file directory
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating file directory: {filedir} for the file: {filename}")
    
    # Logic to create the file if it doesn't exist or is empty(prevents overwriting)
    if(not os.path.exists(filepath) or (os.path.getsize(filepath) == 0)):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filepath}, already exists.")