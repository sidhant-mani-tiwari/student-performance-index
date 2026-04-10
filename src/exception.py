import sys
from src.logger import logging

def error_message_details(error, error_details:sys): # type: ignore
    
    """
    Function to extract and format detailed error information including 
    the file name and line number where the exception occurred.
    """
    
    # extracting the traceback
    _, _, exc_tb = error_details.exc_info()
    
    # extracting the filename
    filename = exc_tb.tb_frame.f_code.co_filename # type: ignore
    
    # extracting the line number
    line_number = exc_tb.tb_lineno # type: ignore
    
    error_message = (
        f"Error occurred in script: [{filename}] "
        f"at line number: [{line_number}] "
        f"with message: [{str(error)}]"
    )
    return error_message

class CustomException(Exception):
    """
    A custom exception class that inherits from the base Exception class
    to provide more descriptive error reporting for our ML pipeline.
    """
    def __init__(self, error_message, error_details:sys): # type: ignore
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_details=error_details)
        logging.error(self.error_message)
    def __str__(self):
        return self.error_message