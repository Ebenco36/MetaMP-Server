import os

def clear_file_content(file_path):
    """
    Clears the content of the specified file.

    Parameters:
    file_path (str): The path to the file to be cleared.

    Returns:
    bool: True if the file was successfully cleared, False if the file does not exist.
    """
    if os.path.exists(file_path):
        with open(file_path, 'w') as file:
            pass
        return True
    else:
        return False