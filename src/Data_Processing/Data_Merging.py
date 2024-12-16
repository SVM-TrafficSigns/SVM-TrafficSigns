import os
import shutil
from PIL import Image
from lxml import etree  # Using lxml to handle XML

# Path to the main directory
base_img_dir = '/home/beboy/Desktop/projects/Traffic Sign Detection/datasets/final_dataset/images'
base_anno_dir = '/home/beboy/Desktop/projects/Traffic Sign Detection/datasets/final_dataset/annotations'

# List of directories to be added
new_dirs = [
    '/home/beboy/Desktop/projects/Traffic Sign Detection/datasets/dataset_daxoa/crosswalk',
    '/home/beboy/Desktop/projects/Traffic Sign Detection/datasets/dataset_daxoa/no_parking',
    '/home/beboy/Desktop/projects/Traffic Sign Detection/datasets/dataset_daxoa/stop',
    '/home/beboy/Desktop/projects/Traffic Sign Detection/datasets/dataset_daxoa/turn_right'
]

# Get the current number of files to continue the numbering
current_img_count = len(os.listdir(base_img_dir))  # Current number of files in the main directory (877)
current_anno_count = len(os.listdir(base_anno_dir))  # Current number of annotation files in the main directory (877)

# Check if the number of image files matches the number of XML files
if current_img_count != current_anno_count:
    print("Error: The number of image files and XML files in the main directory does not match!")
else:
    print(f"Preparing to merge datasets, starting from file {current_img_count}.")

    for new_dir in new_dirs:
        # Path to the subfolders of each new folder
        img_sub_dir = os.path.join(new_dir, 'images (.jpg)')
        anno_sub_dir = os.path.join(new_dir, 'annotations (.xml)')

        # Iterate through each image and XML file
        for img_file in sorted(os.listdir(img_sub_dir)):
            if img_file.endswith(('.jpg', '.png')):  # Check for image file types

                # Create new filenames
                new_img_filename = f"road{current_img_count}.png"  # No additional increment
                new_anno_filename = f"road{current_img_count}.xml"

                # Source paths for the image and XML files
                img_src_path = os.path.join(img_sub_dir, img_file)
                anno_src_path = os.path.join(anno_sub_dir, img_file.replace('.jpg', '.xml').replace('.png', '.xml'))

                # Destination paths in the main directory
                img_dst_path = os.path.join(base_img_dir, new_img_filename)
                anno_dst_path = os.path.join(base_anno_dir, new_anno_filename)

                # Convert image to PNG (if necessary)
                img = Image.open(img_src_path)
                img = img.convert('RGB')
                img.save(img_dst_path, format='PNG')

                # Copy and rename the image file
                shutil.copy(img_src_path, img_dst_path)

                # Open and update the XML file
                tree = etree.parse(anno_src_path)  # Open the XML file
                root = tree.getroot()  # Get the root of the XML file

                # Find and modify the value in the <filename> tag
                filename_tag = root.find(".//filename")  # Find the <filename> tag
                if filename_tag is not None:
                    filename_tag.text = new_img_filename  # Update the <filename> value

                # Save the updated XML file
                tree.write(anno_dst_path)

                print(f"Added: {new_img_filename} and {new_anno_filename}")

                # Increment counters after merging a file
                current_img_count += 1
                current_anno_count += 1

    print("Dataset merging process completed.")