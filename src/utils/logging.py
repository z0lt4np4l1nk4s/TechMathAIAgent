from datetime import datetime
from common.config import LOGGING_ENABLED

def log_info(message: str):
    """Logs a general informational message."""
    _log_message("INFO", message)

def log_warning(message: str):
    """Logs a warning message indicating a potential issue."""
    _log_message("WARNING", message)

def log_error(message: str):
    """Logs an error message for failed operations or exceptions."""
    _log_message("ERROR", message)

def log_success(message: str):
    """Logs a success message for successfully completed tasks."""
    _log_message("SUCCESS", message)

def _log_message(level: str, message: str):
    """
    Internal helper to format and print log messages to the console.
    
    Checks the LOGGING_ENABLED flag from the configuration to decide 
    whether to output the log or suppress it.
    """
    if not LOGGING_ENABLED:
        return
    
    # Get the current timestamp in YYYY-MM-DD HH:MM:SS format
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Print the formatted log string
    print(f"{date} - {level}: {message}")