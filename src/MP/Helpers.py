import os

def extract_model_and_dim_reduction(filename):
    """Extract the model name and dimensionality reduction technique from the filename.

    Args:
        filename (str): The filename to extract information from.

    Returns:
        tuple: A tuple containing the model name and dimensionality reduction technique.
    """
    # Remove the file extension
    base_name = os.path.splitext(filename)[0]

    # Split the base name by '__'
    parts = base_name.split('__')

    if len(parts) == 2:
        model_name, dim_reduction = parts
    else:
        model_name, dim_reduction = None, None

    return model_name, dim_reduction

def get_joblib_files_and_splits(directory):
    """Get all .joblib files in the directory and extract model and dimensionality reduction information.

    Args:
        directory (str): The path to the directory to search.

    Returns:
        tuple: Two lists, one with model names and one with dimensionality reduction techniques.
    """
    joblib_files = [f for f in os.listdir(directory) if f.endswith('.joblib')]
    model_names = []
    dim_reductions = []
    
    for filename in joblib_files:
        model_name, dim_reduction = extract_model_and_dim_reduction(filename)
        if model_name and dim_reduction:
            model_names.append(model_name)
            dim_reductions.append(dim_reduction)
    
    return model_names, dim_reductions