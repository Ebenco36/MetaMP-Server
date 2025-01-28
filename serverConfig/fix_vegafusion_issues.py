import os
import vegafusion


def update_vegafusion_runtime():
    """
    Updates the 'runtime.py' file in the vegafusion package by replacing:
    'imported_inline_datasets[name] = Table(pa.from_pandas(inner_value))'
    with:
    'imported_inline_datasets[name] = pa.Table.from_pandas(inner_value)'.

    This function is designed to be used as part of a Docker entrypoint or setup script.

    Returns:
        str: A message indicating the success or failure of the operation.
    """
    try:
        # Locate the vegafusion runtime.py file
        vegafusion_dir = os.path.dirname(vegafusion.__file__)
        runtime_file = os.path.join(vegafusion_dir, "runtime.py")

        # Verify the file exists
        if not os.path.exists(runtime_file):
            return f"Error: {runtime_file} not found."

        # Read the file content
        with open(runtime_file, "r") as file:
            content = file.read()

        # Define the line to replace and its replacement
        target_line = "imported_inline_datasets[name] = Table(pa.from_pandas(inner_value))"
        replacement_line = "imported_inline_datasets[name] = pa.Table.from_pandas(inner_value)"

        # Check if the target line exists
        if target_line not in content:
            return f"Error: Target line not found in {runtime_file}. No changes made."

        # Replace the target line
        updated_content = content.replace(target_line, replacement_line)

        # Write the changes back to the file
        with open(runtime_file, "w") as file:
            file.write(updated_content)

        return f"Successfully updated {runtime_file}."

    except Exception as e:
        return f"Error occurred while updating {runtime_file}: {e}"


# Entry point for the Docker container
if __name__ == "__main__":
    message = update_vegafusion_runtime()
    print(message)
