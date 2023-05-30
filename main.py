import cv2
import csv
import os
import argparse
import json

# Path to the folder containing the frames
# frames_folder = r'E:\Major_project_main\Main\test_data\Test_Data'


# Create the argument parser
parser = argparse.ArgumentParser(description="Generate crooped image and ball_info.csv for frames in a file.")
# Add the file name argument
parser.add_argument("file", type=str, help="Path to the image folder")
# Parse the command-line arguments
args = parser.parse_args()
# Get the file name from the arguments
frames_folder = args.file
# Path to the CSV file
csv_file = os.path.join(frames_folder,"ball_info.csv")

# Temporary folder to save the images
temp = "temp"
temp_folder = os.path.join(frames_folder,temp)
# Output video file name
output_video = os.path.join(frames_folder,"output_video1.mp4")

# Create the temporary folder if it doesn't exist
os.makedirs(temp_folder, exist_ok=True)
# Read the CSV file
with open(csv_file, 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip the header row if present


    # Iterate over each row in the CSV file
    prev_frame_circles = [] 
    for row in csv_reader:
        frame_name = row[0]
        x_center = int(float(row[1]))
        y_center = int(float(row[2]))
        radius = int(float(row[3]))

        # Read the frame image
        frame_path = os.path.join(frames_folder, frame_name)
        frame = cv2.imread(frame_path)

        
        # Draw the circle for the current frame
        cv2.circle(frame, (x_center, y_center), radius, (0, 255, 0), -1)
        # print()
        # Draw the circles from previous frames
        for circle in prev_frame_circles:
            prev_x, prev_y, prev_radius = circle
            print(circle)
            cv2.circle(frame, (prev_x, prev_y), prev_radius, (0, 255, 0), -1)
        # Add current circle to the list of previous frame circles
        prev_frame_circles.append([x_center, y_center, radius])

        # Save the modified frame image in the temporary folder
        output_path = os.path.join(temp_folder, frame_name)
        cv2.imwrite(output_path, frame)



    # Save the last frame with the whole trajectory as an image
    last_frame_path = os.path.join(frames_folder, f"Out_{frame_name}")
    cv2.imwrite(last_frame_path, frame)

# Convert the images in the temporary folder to a video
frame_rate = 10  # Adjust the frame rate as needed

# Get the dimensions of the first frame
first_frame_path = os.path.join(temp_folder, os.listdir(temp_folder)[0])
first_frame = cv2.imread(first_frame_path)
frame_width, frame_height = first_frame.shape[1], first_frame.shape[0]

# Initialize the video writer
output = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height))

# Iterate over the images in the temporary folder to write them to the output video
image_files = sorted(os.listdir(temp_folder))
for image_file in image_files:
    image_path = os.path.join(temp_folder, image_file)
    frame = cv2.imread(image_path)
    output.write(frame)

# Release the video writer
output.release()

for image_file in image_files:
    image_path = os.path.join(temp_folder, image_file)
    os.remove(image_path)
os.rmdir(temp_folder)


# Read camera parameters from the JSON file
with open(r"E:\Major_project_main\Main\camera_params.json", "r") as f:
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

# folder_path = "E:/Major_project_main/Main/test_data/MVl_3208_1"
csv_file = [file for file in os.listdir(frames_folder) if file.endswith('.csv') and 'ball_info' in file][0]
if len(csv_file) == 0:
    print("Given folder does not contain required ball_info.csv file")

csv_path = os.path.join(frames_folder, csv_file)
json_file = os.path.splitext(csv_file)[0] + '_3d.json'
json_path = os.path.join(frames_folder, json_file)
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