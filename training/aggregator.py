import os
import json

def append_json_files(folder_path, output_file):
    aggregated_data = []

    # Iterate through files in the folder
    for file_name in os.listdir(folder_path):
        # Check if the file matches the naming pattern
        if file_name.startswith("cotagent_") and file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)

            try:
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)
                    aggregated_data.append(data)
            except json.JSONDecodeError:
                print(f"Error decoding JSON in file: {file_name}")
            except Exception as e:
                print(f"Error reading file {file_name}: {e}")

    # Write aggregated data to the output file
    try:
        with open(output_file, 'w') as output_json_file:
            json.dump(aggregated_data, output_json_file, indent=4)
        print(f"Aggregated data written to {output_file}")
    except Exception as e:
        print(f"Error writing to output file: {e}")

# Example usage
folder_path = "./training/measurements"
output_file = "aggregated_data.json"
append_json_files(folder_path, output_file)
