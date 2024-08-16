import os
import re


def delete_non_matching_files(directory_path):
    # Define the regex pattern for files that start with 'P' or 'p' followed by numbers
    pattern = re.compile(r"^[Pp]\d+\..+$")

    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        # Full path to the file
        file_path = os.path.join(directory_path, filename)

        # Check if the path is a file and if it doesn't match the pattern
        if os.path.isfile(file_path) and not pattern.match(filename):
            print(f"Deleting file: {filename}")
            os.remove(file_path)


if __name__ == "__main__":
    # Example usage
    directory = input("Enter the directory path: ")
    delete_non_matching_files(directory)
