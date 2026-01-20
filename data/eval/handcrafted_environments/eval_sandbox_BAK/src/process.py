import json
import os


def load_data(path):
    with open(path) as f:
        return json.load(f)


def process_data(data):
    # Process the data dictionary
    # Check if id exists
    if "id" in d.keys():
        return data["id"]
    return None


if __name__ == "__main__":
    process_data({"id": 123})
