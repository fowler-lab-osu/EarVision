#!/usr/bin/env python3.7

#########################################################
#  EarVision
#
#  Copyright 2020
#
#  Cedar Warman
#  Michaela E. Buchanan
#  Christopher M. Sullivan
#  Justin Preece
#  Pankaj Jaiswal
#  John Folwer
# 
#
#  Department of Botany and Plant Pathology
#  Center for Genome Research and Biocomputing
#  Oregon State University
#  Corvallis, OR 97331
#
#  fowlerj@science.oregonstate.edu
#
#  This program is not free software; you can not redistribute it and/or
#  modify it at all.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
#########################################################


"""

data_prep.py
Cedar Warman's code consolidated into one tool by Michaela Buchanan

Usage:

ex: 
    python data_prep.py -i data/training/2019/ -o ./output/data -m models/research/

Options:
    -i : path to data input directory (ie .png images)
    -o : (optional) path to output directory 
    -m : path to model directory (ie seed_models/research)
    -n : (optional) ratio of training to validation data (default is 0.7)
    -q : (optional) if True, enables quiet mode which reduces amount of terminal output
...

"""

import pyfiglet
import sys
import os
import io
import cv2
from copy import deepcopy
from pathlib import Path
import numpy as np
import argparse
from lxml import etree
from PIL import Image

# access scripts folder for imports
sys.path.insert(1, './scripts')

"""
Import functions from various scripts
"""
# image cropping functions
from crop_annotations_and_images import *

# csv from xml generation function
from xml_to_csv import *

# tfrecord file generation functions
from generate_tfrecord import *

"""
Setting up argument passing
"""

parser = argparse.ArgumentParser(description='Crop annotations and images')
parser.add_argument('-i',
                    '--input_dir',
                    type=str,
                    help=('Path to input directory'),
                    required=True)
parser.add_argument('-o',
                    '--output_dir',
                    type=str,
                    help=('Path to output directory'),
                    default="./output/data")
parser.add_argument('-m',
                    '--model_dir',
                    type=str,
                    help=('Path to model directory (ie tensorflow/research)'),
                    default='./seed_models/research/')
parser.add_argument('-n',
                    '--image_split_num',
                    type=int,
                    default=3,
                    help=('Number of subdivisions to crop into.'))
parser.add_argument('-r',
                    '--train_validation_ratio',
                    type=float,
                    default=0.7,
                    help=('Ratio of training to validation data (0.7 is default)'))
parser.add_argument('-q',
                    '--quiet',
                    type=bool,
                    default=False,
                    help=('Will not show as much output text if True'))
args = parser.parse_args()

""" 
Setting up environmental variables for object detection model
"""

# handle / on input path
newPath = args.model_dir

if newPath[0] == '/':
    newPath = newPath[1:]


# create new path
sys.path.append(newPath) 

from object_detection.utils import dataset_util 

"""
====
Main
====
"""

def main():
    """
    Pretty banner (you're welcome Chris)
    """

    ascii_banner = pyfiglet.figlet_format("Seed Project Data Preperation Tool")
    print(ascii_banner)


    """
    crop_annotations_and_images.py 
    """

    # Making the output directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir + "/annotations").mkdir(parents=True, exist_ok=True)
    
    # Setting up some paths
    input_dir = Path(args.input_dir)
    annotations_dir = input_dir / "annotations"

    # Setting up a dictionary to store the final cropped dimensions (the reason
    # I did it like this is that the final dimensions come from the image
    # cropping for loop, but I want to store them in a place that can be pulled
    # from the xml loop. Not sure if this is the most efficient way, but it
    # works). For now I'll just store the x, because the y should all be the
    # same.
    image_dimensions = {}

    print("\nProcessing images...")

    # Cropping the images
    for image_file in os.listdir(input_dir):
        image_path = str(input_dir / image_file)
        if image_path.endswith(".png"):
            if not args.quiet:
                print("Processing image file: " + str(image_file))
            current_image = str(image_file[:-4])

            # Importing and splitting up the image
            image = load_image_into_numpy_array(image_path)
            split_images = split_image_np_array(image, args.image_split_num)

            # Saving the images
            save_split_images(split_images, args.output_dir, current_image)

            # Getting the exact dimensions of the split image (they might vary
            # a littlebased on rounding). This will be used on the xml files to
            # make the dimensions elements reflect the cropped image.
            x_coord_list = []
            for sub_image in split_images:
                x_coord_list.append(sub_image.shape[1])
            image_dimensions[current_image] = x_coord_list

    # Going through each annotation file
    for xml_file in os.listdir(annotations_dir):
        if not args.quiet:
            print("Processing xml file: " + str(xml_file))
        xml_path = str(annotations_dir / xml_file)
        output_tree = split_annotations(xml_path, 3, image_dimensions) 
        
        # Printing out the subtrees
        tree_number_print = 1
        for output in output_tree:
            tree_name_output_string = (args.output_dir + "/annotations/" + 
                xml_file[:-4] + "_s" + str(tree_number_print) + ".xml")

            xml_string = etree.tostring(output, pretty_print=True)

            with open(tree_name_output_string, 'wb') as f:
                f.write(xml_string)

            tree_number_print += 1

    """
    xml_to_csv.py 
    """

    image_path = os.path.join(args.output_dir, 'annotations')
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv(args.output_dir + "/test_output.csv", index=None)
    print('\nSuccessfully converted xml to csv.\n')

    """
    split_labels.py 
    """

    # Importing the csv input
    full_labels = pd.read_csv(args.output_dir + "/test_output.csv")
    # Grouping by image name
    gb = full_labels.groupby('filename')

    # Splitting the input into a list of annotations, by image
    grouped_list = [gb.get_group(x) for x in gb.groups]

    # Figuring out the total number of training and validation images
    total_images = len(grouped_list)

    training_num = int(round(total_images * args.train_validation_ratio))

    # Splitting the dataframe
    train_index = np.random.choice(len(grouped_list), size=training_num, replace=False)
    val_index = np.setdiff1d(list(range(total_images)), train_index)

    train = pd.concat([grouped_list[i] for i in train_index])
    val = pd.concat([grouped_list[i] for i in val_index])

    # Printing out the lists
    train_output_path = args.output_dir + '/train_labels.csv'
    validation_output_path = args.output_dir + '/val_labels.csv'

    train.to_csv(train_output_path, index=None)
    val.to_csv(validation_output_path, index=None)

    """
    generate_tfrecord.py for training
    """
    writer = tf.io.TFRecordWriter(args.output_dir + "/train.record")
    path = args.output_dir
    examples = pd.read_csv(args.output_dir + "/train_labels.csv")
    grouped = split(examples, 'filename')

    print("Creating TFrecords file...\n")

    for group in grouped:
        tf_example = create_tf_example(group, path, dataset_util, args.quiet)
        writer.write(tf_example.SerializeToString())

    writer.close()

    """
    generate_tfrecord.py for validation
    """
    writer = tf.io.TFRecordWriter(args.output_dir + "/val.record")
    path = args.output_dir
    examples = pd.read_csv(args.output_dir + "/val_labels.csv")
    grouped = split(examples, 'filename')

    print("Creating TFrecords file...\n")

    for group in grouped:
        tf_example = create_tf_example(group, path, dataset_util, args.quiet)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print("-------------------------------------------------------------------------------")
    print('Successfully created the TFRecords: {}'.format(args.output_dir) + "\n")

if __name__ == "__main__":
    main()
