import sys
from io import StringIO
import numpy as np
import pandas as pd
from utils.logging import log_error

def execute_python_code(code):
    """
    Executes a Python code string and captures any standard output (print statements).
    This is a basic execution wrapper.
    """
    # Define a restricted global environment for the executed code
    global_vars = {
        "__builtins__": __builtins__,
        "np": np,
        "numpy": np,
        "pd": pd,
        "pandas": pd
    }

    # Backup the original stdout to restore it later
    old_stdout = sys.stdout
    # Redirect stdout to a string buffer to catch printed results
    redirected_output = sys.stdout = StringIO()
    
    try:
        # Execute the provided string as Python code
        exec(code, global_vars)
    except Exception as e:
        # Log the error using our custom logger and notify the user
        log_error(f"An error occurred during code execution: {e}")
        return f"Greška u izvršavanju: {e}"
    finally:
        # Always restore the original stdout, even if an exception occurs
        sys.stdout = old_stdout
        
    return redirected_output.getvalue()

def silent_execute(code):
    """
    An advanced execution function that attempts to intelligently find the 
    result of the code, checking for a 'result' variable, printed output, 
    or the last modified variable.
    """
    # 1. Environment Preparation: Inject necessary numerical libraries
    global_vars = {
        "__builtins__": __builtins__,
        "np": np,
        "numpy": np,
        "pd": pd,
        "pandas": pd
    }
    # Local variable dictionary to capture changes after execution
    locs = {}

    # 2. Redirect standard output to capture print() calls from the model
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()

    try:
        # Execute the code; global_vars holds libs, locs captures user-defined variables
        exec(code, global_vars, locs)
        
        # Restore stdout immediately after execution
        sys.stdout = old_stdout
        captured_print = redirected_output.getvalue().strip()

        # 3. RESULT DETERMINATION LOGIC (Priority based):
        
        # Priority A: Check if the model explicitly defined a 'result' variable
        if 'result' in locs:
            return locs['result']
        
        # Priority B: If no 'result' variable, check if the model used print()
        if captured_print:
            return captured_print
            
        # Priority C: Fallback to the last variable created by the model
        # Filter out internal or injected variables to avoid returning library objects
        user_vars = {k: v for k, v in locs.items() if k not in ['__builtins__']}
        if user_vars:
            last_val = list(user_vars.values())[-1]
            return last_val

        return "Kod izvršen (nema ispisa ili varijable 'result')."

    except Exception as e:
        # Critical: Restore stdout even during a crash to prevent freezing the main app
        sys.stdout = old_stdout
        log_error(f"An error occurred during silent code execution: {e}")
        return f"Greška u izvršavanju: {e}"