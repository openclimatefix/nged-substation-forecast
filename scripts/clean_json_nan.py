import os
import re


def clean_json_files(directory):
    """
    Replaces 'NaN' with 'null' in all .json files in the given directory.
    """
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            print(f"Cleaning {filepath}...")

            # Read the file content
            with open(filepath, "r") as f:
                content = f.read()

            # Replace 'NaN' with 'null'
            # Using regex to ensure we only replace 'NaN' as a standalone value
            # and not part of another word.
            cleaned_content = re.sub(r"\bNaN\b", "null", content)

            # Write the cleaned content back to the file
            with open(filepath, "w") as f:
                f.write(cleaned_content)

            print(f"Finished cleaning {filepath}")


if __name__ == "__main__":
    target_dir = "data/NGED/from_sharepoint/OneDrive_1_4-8-2026/1451606400000_1774512000000/"
    clean_json_files(target_dir)
