import os
import logging
from datetime import datetime

def setup_logger():
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Configure root logger with time and process id
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create console handler with a higher log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create file handler which logs even debug messages
    file_handler = logging.FileHandler(f'logs/app-{datetime.now().strftime("%Y%m%d-%H%M%S")}.log')
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatters and add them to handlers
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - [%(process)d] - %(name)s - %(message)s')
    
    console_handler.setFormatter(console_format)
    file_handler.setFormatter(file_format)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


