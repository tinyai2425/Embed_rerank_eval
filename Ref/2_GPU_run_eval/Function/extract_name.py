import os
import re

def parse_project_base_and_filename(input_path):
    abs_path = os.path.abspath(input_path)
    project_base = os.path.dirname(abs_path)
    file_name = os.path.basename(abs_path)
    return project_base, file_name

def extract_last_number_from_filename(filename):
    match = re.search(r"-([0-9]+)\.[a-zA-Z0-9]+$", filename)
    if not match:
        raise ValueError(f"❌ Cannot extract trailing number from filename: {filename}")
    return int(match.group(1))