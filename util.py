from datetime import datetime
import os
from typing import Optional


def generate_file_name(suffix: str, folder: Optional[str] = ".") -> str:
    """
    Generate a dated prefix file name with a given suffix.

    Args:
        suffix (str): The suffix for the file name. It should include the file extension (e.g., '.txt', '.csv', '.jpg').
        folder (Optional[str]): The folder where the file will be created. If None, defaults to the current directory.

    Returns:
        str: The generated file name.
    """
    if not suffix:
        raise ValueError(
            "Suffix must not be empty. Please provide a valid suffix or file extension (e.g., '.txt', '.csv', '.jpg')."
        )
    if folder is None:
        folder = "."
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    if not folder.endswith("/"):
        folder += "/"
        
    date_prefix = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not suffix.startswith("."):
        suffix = "." + suffix

    # if file name already exists, append a number to the suffix
    counter = 1
    while True:
        file_name = f"{folder}{date_prefix}_{counter}{suffix}"
        if not os.path.exists(file_name):
            break
        counter += 1
    return file_name
