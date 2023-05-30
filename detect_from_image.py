import numpy as np
import argparse
import os
import tensorflow as tf
from PIL import Image
from io import BytesIO
import glob
import csv
import matplotlib.pyplot as plt

from utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model

def save_detection(csv_path,output_dict ):
    with open(csv_path, 'w', newline='') as csv_file:
        # print("csv file opened")
        writer = csv.writer(csv_file)
        writer.writerow(['Image_name','Class', 'Score', 'YMin', 'XMin', 'YMax', 'XMax'])
        for j in range(output_dict['num_detections']):
            image_name = output_dict['image_name']
            class_id = output_dict['detection_classes'][j]
            class_name = category_index[class_id]['name']
            score = output_dict['detection_scores'][j]
            ymin, xmin, ymax, xmax = output_dict['detection_boxes'][j]
            writer.writerow([image_name,class_name, score, ymin, xmin, ymax, xmax])

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: a file path (this can be local or on colossus)

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(model, image,i_name):
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # Custumize
    output_dict['image_name'] = i_name

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def run_inference(model, category_index, image_path, save_csv = False):
    print("inside inference")
    if os.path.isdir(image_path):
        image_paths = []
        print("image path PASS")
        # Creating output folder in outputs directory
        image_folder = os.path.basename(image_path)
        output_folder = r"E:\Major_project_main\Main\outputs"
        output_folder_path = os.path.join(output_folder,image_folder)

        if not os.path.exists(output_folder_path):
            print(f"Creating Output folder {output_folder_path}")
            os.mkdir(output_folder_path)

        csv_file_path = os.path.join(output_folder_path,"{}_detection.csv".format(image_folder))
        for file_extension in ('*.png', '*.jpg'):
            image_paths.extend(glob.glob(os.path.join(image_path, file_extension)))

        """add iterator here"""

        for i_path in image_paths:
            image_np = load_image_into_numpy_array(i_path)
            # Actual detection.
            i_name = os.path.basename(i_path)[:-4]
            output_dict = run_inference_for_single_image(model, image_np, os.path.basename(i_path))
            # print(type(output_dict))
            # print(output_dict)
            # Save Detection in csv file
            if save_csv:
                save_detection(csv_file_path,output_dict )

            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks_reframed', None),
                use_normalized_coordinates=True,
                line_thickness=8)
            """The existing plt lines do not work on local pc as they are not setup for GUI
                Use plt.savefig() to save the results instead and view them in a folder"""
            plt.imshow(image_np)
            # plt.show()
            plt.savefig(os.path.join(output_folder_path,"{}_output.png".format(i_name)))  # make sure to make an outputs folder
            # print(f"image is saved in {output_folder_path}")

    # else:
    #     image_np = load_image_into_numpy_array(image_path)
    #     # Actual detection.
    #     output_dict = run_inference_for_single_image(model, image_np)
    #     # Visualization of the results of a detection.
    #     vis_util.visualize_boxes_and_labels_on_image_array(
    #         image_np,
    #         output_dict['detection_boxes'],
    #         output_dict['detection_classes'],
    #         output_dict['detection_scores'],
    #         category_index,
    #         instance_masks=output_dict.get('detection_masks_reframed', None),
    #         use_normalized_coordinates=True,
    #         line_thickness=8)
    #     plt.imshow(image_np)
    #     plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect objects inside webcam videostream')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model Path')
    parser.add_argument('-l', '--labelmap', type=str, required=True, help='Path to Labelmap')
    parser.add_argument('-i', '--image_path', type=str, required=True, help='Path to image (or folder)')
    args = parser.parse_args()

    detection_model = load_model(args.model)
    category_index = label_map_util.create_category_index_from_labelmap(args.labelmap, use_display_name=True)

    run_inference(detection_model, category_index, args.image_path)

# Command to start script
#python .\detect_from_image.py -m .\inference_graph3\saved_model -l .\labelmap.pbtxt -i .\my_data\MVI_3218_1
