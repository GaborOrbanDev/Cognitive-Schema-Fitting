import os
import json

def group_by_label_json_files(label, folder_path, output_file):
    aggregated_data = []

    # Iterate through files in the folder
    for file_name in os.listdir(folder_path):
        # Check if the file matches the naming pattern
        if file_name.startswith(label) and file_name.endswith(".json"):
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


def merge_into_one_file(files: list[str], output_file: str):
    aggregated_data = []

    for file_path in files:
        try:
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
                aggregated_data.extend(data)
        except json.JSONDecodeError:
            print(f"Error decoding JSON in file: {file_path}")
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    # Write aggregated data to the output file
    try:
        with open(output_file, 'w') as output_json_file:
            json.dump(aggregated_data, output_json_file, indent=4)
        print(f"Aggregated data written to {output_file}")
    except Exception as e:
        print(f"Error writing to output file: {e}")


merge_into_one_file(
    [
        "aggregated_data_cot.json",
        "aggregated_data_cot_sr.json",
        "aggregated_data_cot_sc.json",
        "aggregated_data_spp.json",
        "aggregated_data_tot.json",
    ], 
    "all_aggregated_data.json"
)