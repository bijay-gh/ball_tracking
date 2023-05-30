import os
import pandas as pd
import tensorflow as tf
from PIL import Image

def create_tf_example(row, image_folder):
    image_path = os.path.join(image_folder, row['name'])
    with open(image_path, 'rb') as f:
        encoded_image_data = f.read()

    image = Image.open(image_path)
    width, height = image.size

    filename = row['name'].encode('utf8')
    xmins = [row['xmin'] / width]
    xmaxs = [row['xmax'] / width]
    ymins = [row['ymin'] / height]
    ymaxs = [row['ymax'] / height]
    classes_text = [row['class'].encode('utf8')]
    classes = [1]  # Assign an integer label to the class, e.g., 1

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image_data])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'jpeg'])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
    }))
    return tf_example

def csv_to_tfrecord(csv_path, image_folder, output_path):
    df = pd.read_csv(csv_path)
    with tf.io.TFRecordWriter(output_path) as writer:
        for _, row in df.iterrows():
            tf_example = create_tf_example(row, image_folder)
            writer.write(tf_example.SerializeToString())

if __name__ == '__main__':
    csv_path = r'E:\TF2\models\research\object_detection\final_data\train.csv'  # Replace with the path to your CSV file
    image_folder = r'E:\TF2\models\research\object_detection\final_data\train'  # Replace with the path to your image folder
    output_path = r'E:\TF2\models\research\object_detection\final_data\train.tfrecord'  # Replace with the desired output path
    csv_to_tfrecord(csv_path, image_folder, output_path)