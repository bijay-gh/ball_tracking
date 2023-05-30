import os
import csv
import json
import argparse

# folder_path = "E:/Major_project_main/Main/test_data/MVl_3208_1"

# Create the argument parser
parser = argparse.ArgumentParser(description="Generate crooped image and ball_info.csv for frames in a file.")
# Add the file name argument
parser.add_argument("file", type=str, help="Path to the image folder")
# Parse the command-line arguments
args = parser.parse_args()
# Get the file name from the arguments
folder_path = args.file

csv_file = [file for file in os.listdir(folder_path) if file.endswith('.csv') and 'ball_info' in file][0]
if len(csv_file) == 0:
    print("Given folder does not contain required ball_info.csv file")


# Read camera parameters from the JSON file
with open("./camera_params.json", "r") as f:
    camera_params = json.load(f)

fx = camera_params["intrinsic"]["fx"]
fy = camera_params["intrinsic"]["fy"]
ox = camera_params["intrinsic"]["ox"]
oy = camera_params["intrinsic"]["oy"]

# Function to calculate x, y, z coordinates of the ball in the camera coordinate system
def calculate_coordinates(x_center, y_center, radius, ball_radius_mm=182):
    z = (fx * ball_radius_mm) / radius
    x = (x_center - ox) * z / fx
    y = (y_center - oy) * z / fy
    return x, y, z

csv_path = os.path.join(folder_path, csv_file)
json_file = os.path.splitext(csv_file)[0] + '_3d.json'
json_path = os.path.join(folder_path, json_file)
# print(json_path)

frame_data = []
with open(csv_path, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader: 
        x_center = float(row['x_center'])
        y_center = float(row['y_center'])
        radius = float(row['radius'])
        x, y, z = calculate_coordinates(x_center, y_center, radius)
        data = {
            "frame_name": row["filename"],
            "x": x,
            "y":y,
            "z":z 
        }
        frame_data.append(data)
with open(json_path, 'w') as file:
    json.dump(frame_data, file, indent=4)

