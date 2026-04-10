import logging
import os
from datetime import datetime

# ----------------------------------------------------------
# STEP 1: Create a timestamped log filename
# Every run of your pipeline gets its own log file.
# Format: MM_DD_YYYY_HH_MM_SS.log
# Example: 04_10_2026_14_35_22.log
# ----------------------------------------------------------
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# ----------------------------------------------------------
# STEP 2: Build the path where log files will be saved
# logs/
# └── 04_10_2026_14_35_22.log
# ----------------------------------------------------------
logs_dir  = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# ----------------------------------------------------------
# STEP 3: Configure the logging system
# This is the one-time setup that tells Python's logging
# module HOW and WHERE to record messages.
# ----------------------------------------------------------
logging.basicConfig(

    # Write logs to our timestamped file
    filename=LOG_FILE_PATH,

    # Format of each log line:
    # [timestamp] line_number root - SEVERITY - message
    # Example:
    # [2026-04-10 14:35:22,123] 45 root - INFO - Read dataset as dataframe
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",

    # Only record INFO and above — ignore DEBUG in production
    level=logging.INFO,
)