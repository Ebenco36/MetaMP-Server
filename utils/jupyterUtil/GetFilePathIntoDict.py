import os


def getFiles(directory_path = "/Users/AwotoroE-Dev/Desktop/HH/dimension_reduction_datasets"):
    # Initialize an empty dictionary to store the file names (without extensions) and paths
    file_dict = {}
  
    # Check if the directory exists
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        # List all files in the directory
        for filename in os.listdir(directory_path):
            # Get the full file path
            file_path = os.path.join(directory_path, filename)
            # Check if it's a file (not a directory)
            if os.path.isfile(file_path):
                # Remove the file extension from the filename
                file_name_without_extension, file_extension = os.path.splitext(filename)
                # Add the file name without extension and path to the dictionary
                file_dict[file_name_without_extension] = file_path
    
        # Now, you have a dictionary with file names (without extensions) as keys and file paths as values
        return file_dict
    else:
        return {}


