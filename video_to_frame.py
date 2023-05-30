
import argparse
import cv2
import os

# Create an argument parser
parser = argparse.ArgumentParser(description='Extract frames from a video and create separate subdir(as name of video file) in output dir for each input file in input dir')
parser.add_argument('--input_dir', help='input video file directory')
parser.add_argument('--output_dir', help='output directory for frames')
args = parser.parse_args()

# Read the input video file
input_dir = args.input_dir
output_dir = args.output_dir

# Check if the input file exists
if not os.path.exists(input_dir):
    print("Error: input file does not exist.")
    exit()
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

input_dirs = []
file_name = os.listdir(input_dir)
for file in file_name:
    src_path = os.path.join(input_dir, file)
    input_dirs.append(src_path)

for dir in input_dirs:
    # Extract the name of the input file without the extension
    file_name = os.path.splitext(os.path.basename(dir))[0]
    output_subdir = os.path.join(output_dir, file_name)

    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)
    # Open the video file
    video_capture = cv2.VideoCapture(dir)

    # Initialize a counter for the frames
    frame_count = 0

    # Loop through the video and extract each frame
    while True:
        # Read a single frame from the video
        ret, frame = video_capture.read()

        # If the frame cannot be read, break out of the loop
        if not ret:
            break
        # Increment the frame counter
        frame_count += 1

        # Write the frame to a file in the specified output directory
        frame_file = os.path.join(output_subdir, f'{file_name}_frame{frame_count}.jpg')
        cv2.imwrite(frame_file, frame)

    # Release the video capture object and close any windows
    video_capture.release()
    cv2.destroyAllWindows()
