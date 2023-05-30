import pandas as pd
import numpy as np
import os,csv
import numpy as np
import xml.etree.ElementTree as ET
import re
import argparse
# Create the argument parser
parser = argparse.ArgumentParser(description="Generate csv file from the xml file")
# Add the file name argument
parser.add_argument("file", type=str, help="Path to the folder containing xml file")
# Parse the command-line arguments
args = parser.parse_args()
# Get the file name from the arguments
folder_name = args.file


def xml_to_csv(folder_name):
    for item in os.listdir(folder_name):
        # print(item)
        if item.endswith(".csv"):
            raise TypeError("Already contain csv file. Proceed next....")
        if item.endswith('.xml'):
                data = []
                # Process the XML file
                # print(f"Processing {item}")
                tree = ET.parse(os.path.join(folder_name,item))
                root = tree.getroot()
                
                for image in root.findall('image'):
                        name = image.get('name')
                        width = image.get('width')
                        height = image.get('height')
                        bbox = image.find("box")
                        if bbox is not None:
                            label = bbox.get("label")
                            xtl = bbox.get('xtl')
                            ytl = bbox.get('ytl')
                            xbr = bbox.get('xbr')
                            ybr = bbox.get('ybr')
                        else:
                            xtl = ytl = xbr = ybr = label = np.nan
                    
                        image_data = {
                            'name': name,
                            'width': width,
                            'height':height,
                            'class': label,
                            'xmin': xtl,
                            'ymin': ytl,
                            'xmax': xbr,
                            'ymax': ybr
                        }
                        data.append(image_data)
        # Write the data to the CSV file
        fieldnames = ['name', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        csv_file = os.path.join(folder_name,"annotation.csv")
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
    # Sorting the csv file according to frame number
    df = pd.read_csv(csv_file)
    df.dropna(inplace=True)
    convert_to_two_digits = lambda x: re.sub(r'(\d+)(\.[a-zA-Z]+)$', lambda m: f'{int(m.group(1)):02d}{m.group(2)}', x)
    df["name"] = df['name'].apply(convert_to_two_digits)
    df = df.sort_values(by='name')
    columns_to_convert = ["width", "height","xmin", "ymin", "xmax", "ymax"]
    desired_data_type = int
    # Convert the selected columns to the desired data type
    df[columns_to_convert] = df[columns_to_convert].astype(desired_data_type)
    df.to_csv(csv_file, index=False)

# folder_name = r"E:\Major_project_main\Main\test_data\MVl_3239_2"
# xml_to_csv(folder_name)