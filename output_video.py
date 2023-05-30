import cv2
import csv
import os
import argparse

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

# python outp