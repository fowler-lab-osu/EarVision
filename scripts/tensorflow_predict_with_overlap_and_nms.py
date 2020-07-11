#!/usr/bin/env python3

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
This version will impliment overlap in image subdivisions and non-maximum
suppression. Eventually it will replace the other version, but for now there
are two in case I break this one.

tensorflow_predict.py
Adapted from:
https://colab.research.google.com/github/Tony607/object_detection_demo/blob/master/tensorflow_object_detection_training_colab.ipynb#scrollTo=mz1gX19GlVW7

Usage (note: I've been running this on a Nvidia GPU with Tensorflow 1.12-GPU 
in a conda virtual environment):

bash; conda activate tf1.12-gpu; python tensorflow_predict.py 
    -c <checkpoint_path> 
    -l <label_path> 
    -d <test_image_directory> 
    -o <output_directory>
    -m <object detection model directory>
    -s <min_score_threshold>
    -n <image_split_number>
"""

import os
import glob

import numpy as np
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import csv
import argparse

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# Setting up arguments
parser = argparse.ArgumentParser(description='Splits labels')
parser.add_argument('-c', 
                    '--checkpoint',
                    type=str,
                    help=('Path to frozen detection graph (checkpoint)'))
parser.add_argument('-l',
                    '--labels',
                    type=str,
                    default='/home/bpp/warmanma/warman_nfs0/computer_vision/tensorflow/transfer_learning_4/data/label_map.pbtxt',
                    help=('Path to class label map'))
parser.add_argument('-d',
                    '--test_image_dir',
                    type=str,
                    help=('Path to test image directory'))
parser.add_argument('-o',
                    '--output_path',
                    type=str,
                    help=('Path to output directory'))
parser.add_argument('-m',
                    '--model_path',
                    type=str,
                    help=('Path to object detection model directory'))
parser.add_argument('-s',
                    '--min_score_threshold',
                    type=float,
                    default=0.05,
                    help=('Minimum score threshold for plotting bounding boxes'))
parser.add_argument('-n',
                    '--image_split_num',
                    type=int,
                    default=1,
                    help=('Number of image subdivisions to run the object detection on.'))
parser.add_argument('-w',
                    '--overlap_width',
                    type=int,
                    default=100,
                    help=('Pixel overlap width for image subdivisions.'))
args = parser.parse_args()

""" 
Setting up environmental variables for object detection model
"""

# handle / on input path
newPath = args.model_path

if newPath[0] == '/':
    newPath = newPath[1:]

netsPath = newPath + "/tf_slim"

# create new path
sys.path.append(newPath) 
sys.path.append(netsPath)

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = args.checkpoint

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = args.labels

# If you want to test the code with your images, just add images files to the PATH_TO_TEST_IMAGES_DIR.
PATH_TO_TEST_IMAGES_DIR = args.test_image_dir

assert os.path.isfile(PATH_TO_CKPT)
assert os.path.isfile(PATH_TO_LABELS)
TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, "*.*"))
assert len(TEST_IMAGE_PATHS) > 0, 'No image found in `{}`.'.format(PATH_TO_TEST_IMAGES_DIR)

# Importing the frozen inference graph
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=2, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Loading the images
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    print(im_width)
    print(im_height)
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

# A really complicated fuction to get the split sections, with overlap
def get_splits(image_width, split_number, overlap):

    image_splits = []
    total_image_width = image_width
    overlap_width = overlap

    if args.image_split_num == 1:
        image_splits.append([0, total_image_width]) 
        #print("Lower image split number is: ")
        #print(image_splits[0][0])
        #print("Upper image split number is: ")
        #print(image_splits[0][1])

    # This will be the most used case, as of now, with a split of 3 sub-images.
    # In this case, since a lot of the ear images have significant space on the
    # left and right, I want the center sub-image to not be too big. To avoid
    # this, I'll do the overlaps from the left and right images and leave the
    # center image unchanged.` 
    elif args.image_split_num == 3:
        # Here's the split width if there's no overlap (note: probably will
        # need to do something about rounding errors here with certain image
        # widths).
        no_overlap_width = int(total_image_width / split_number)
        
        # Left split. The left side of the left split will always be zero.
        left_split = []
        left_split.append(0)

        # The other side of the left split will be the width (minus 1 to fix
        # the 0 index start) plus the overlap
        left_split.append(no_overlap_width + overlap_width)
        image_splits.append(left_split)

        # The middle has no overlap in this case
        middle_split = []
        middle_split.append(no_overlap_width)
        middle_split.append(no_overlap_width * 2)
        image_splits.append(middle_split)

        # The right split is the opposite of the left split
        right_split = []
        right_split.append((2 * no_overlap_width) - overlap_width)
        right_split.append(total_image_width)
        image_splits.append(right_split)

        # Test prints
        #print("Left split is: " + str(image_splits[0][0]) + ", " + str(image_splits[0][1]))
        #print("Middle split is: " + str(image_splits[1][0]) + ", " + str(image_splits[1][1]))
        #print("Right split is: " + str(image_splits[2][0]) + ", " + str(image_splits[2][1]))

    else:
        # If the split is not 1 or 3, this more general overlap setup happens,
        # with overlaps on all boundaries.
        no_overlap_width = int(total_image_width / split_number)

        # Left split
        left_split = []
        left_split.append(0)
        left_split.append(no_overlap_width + overlap_width)
        image_splits.append(left_split)

        # Middle splits (the minus 2 is because the left and right sides are
        # handled separately)
        for split_position in range(1, (split_number - 1)): 
            middle_split = []
            left_middle_split = (no_overlap_width * split_position) - overlap_width
            right_middle_split = (no_overlap_width * (split_position + 1)) + overlap_width
            middle_split.append(left_middle_split)
            middle_split.append(right_middle_split)
            image_splits.append(middle_split)

        # Right split
        right_split = []
        right_split.append((no_overlap_width * (split_number - 1)) - overlap_width)
        right_split.append(total_image_width)
        image_splits.append(right_split)

        # Test prints (this print only works for split = 4)
        #print("Left split is: " + str(image_splits[0][0]) + ", " + str(image_splits[0][1]))
        #print("Middle split 1 is: " + str(image_splits[1][0]) + ", " + str(image_splits[1][1]))
        #print("Middle split 1 is: " + str(image_splits[2][0]) + ", " + str(image_splits[2][1]))
        #print("Right split is: " + str(image_splits[3][0]) + ", " + str(image_splits[3][1]))
    return(image_splits)
    

# The fuction that actually splits the images
def split_image(image_np_array, split_list):
    print(image_np_array.shape)
    array_list = []

    for split_nums in split_list:
        left_border = int(split_nums[0])
        right_border = int(split_nums[1])
        print("Borders:")
        print(left_border)
        print(right_border)
        sub_array = image_np_array[:,left_border:right_border,:]
        array_list.append(sub_array)

    return(array_list)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(
                    tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(
                    tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates 
                # to image coordinates and fit the image size.
                real_num_detection = tf.cast(
                    tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                           real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                           real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(
                output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

# This function is for getting out the total number of fluorescent and
# nonfluorescent boxes detected from an output_dict. 
def get_object_counts(output_dict, min_score):
    detection_classes = output_dict['detection_classes']
    detection_scores = output_dict['detection_scores']

    total_fluorescent = 0
    total_nonfluorescent = 0

    for i in range(len(detection_scores)):
        detect_class = detection_classes[i]
        detect_score = detection_scores[i]
        if detect_score > min_score:
            if detect_class == 1:
                total_fluorescent = total_fluorescent + 1
            if detect_class == 2:
                total_nonfluorescent = total_nonfluorescent + 1

    output_list = [total_fluorescent, total_nonfluorescent]
    return(output_list)

# This function is for saving a file with more detailed information about the
# bounding boxes and confidence scores
def get_boxes_and_scores(output_dict, image_name_string):
    image_name = image_name_string
    detection_boxes = output_dict['detection_boxes']
    detection_scores = output_dict['detection_scores']
    detection_classes = output_dict['detection_classes']

    x_min_list = []
    x_max_list = []
    y_min_list = []
    y_max_list = []
    score_list = []
    class_list = []
    name_list = []

    for i in range(len(detection_scores)):
        x_min = detection_boxes[i][1]
        x_max = detection_boxes[i][3]
        y_min = detection_boxes[i][0]
        y_max = detection_boxes[i][2]
        score = detection_scores[i]
        class_num = detection_classes[i]

        x_min_list.append(x_min)
        x_max_list.append(x_max)
        y_min_list.append(y_min)
        y_max_list.append(y_max)
        score_list.append(score)
        class_list.append(class_num)
        name_list.append(image_name)

    output_list = [x_min_list,
        x_max_list,
        y_min_list,
        y_max_list,
        score_list,
        class_list,
        name_list]

    return(output_list)
    

# This function removes the boxes that are near the edges of the splits, in an
# attempt to remove boxes that are only a fraction of a seed.
def remove_edge_boxes(output_dict, list_of_splits, image_position):
    output_dict_boxes_removed = output_dict
    # This is how close the edge of a box can be to the edge of the sub-image
    # before they get deleted
    edge_crop_width = 40
    image_width = list_of_splits[-1][1]
    relative_edge_crop_width = edge_crop_width / image_width
    #print("relative_edge_crop_width is: ")
    #print(relative_edge_crop_width)

    array_counter = 0
    delete_list = []

    # Pulling out the elements of the output_dict that will get modified
    adjusted_boxes = output_dict['detection_boxes']
    adjusted_scores = output_dict['detection_scores']
    adjusted_classes = output_dict['detection_classes']

    # On the leftmost set of boxes, only ones near the right side will be
    # deleted
    if image_position == 0:
        print("\n\nleft image")
        for box in adjusted_boxes:
            xmax = box[3] 
            if xmax > (1 - relative_edge_crop_width):
                # Adding the index to the list of indexes to be deleted
                delete_list.append(array_counter)
            array_counter += 1
        print(len(delete_list))
        
        adjusted_boxes = np.delete(adjusted_boxes, delete_list, 0)
        adjusted_scores = np.delete(adjusted_scores, delete_list, 0)
        adjusted_classes = np.delete(adjusted_classes, delete_list, 0)

    # Rightmost set
    elif image_position == (len(list_of_splits) - 1):
        print("\n\nright image")
        for box in adjusted_boxes:
            xmin = box[1] 
            if xmin < relative_edge_crop_width:
                # Adding the index to the list of indexes to be deleted
                delete_list.append(array_counter)
            array_counter += 1
        print(len(delete_list))
        
        adjusted_boxes = np.delete(adjusted_boxes, delete_list, 0)
        adjusted_scores = np.delete(adjusted_scores, delete_list, 0)
        adjusted_classes = np.delete(adjusted_classes, delete_list, 0)

    # All the middle sets
    else:
        print("\n\nmiddle image")
        for box in adjusted_boxes:
            xmax = box[3] 
            xmin = box[1] 
            if (xmin < relative_edge_crop_width) or (xmax > (1 - relative_edge_crop_width)):
                # Adding the index to the list of indexes to be deleted
                delete_list.append(array_counter)
            array_counter += 1
        print(len(delete_list))
        
        adjusted_boxes = np.delete(adjusted_boxes, delete_list, 0)
        adjusted_scores = np.delete(adjusted_scores, delete_list, 0)
        adjusted_classes = np.delete(adjusted_classes, delete_list, 0)
    
    # Adding the modified arrays back into the output_dict
    print("Original array length: ")
    print(output_dict_boxes_removed['detection_boxes'].shape[0])

    output_dict_boxes_removed['detection_boxes'] = adjusted_boxes
    output_dict_boxes_removed['detection_scores'] = adjusted_scores
    output_dict_boxes_removed['detection_classes'] = adjusted_classes
    output_dict_boxes_removed['num_detections'] = adjusted_boxes.shape[0]

    print("Modified array length: ")
    print(output_dict_boxes_removed['detection_boxes'].shape[0])

    return(output_dict_boxes_removed)


# This function fixes the relative coordinates when splitting an image into
# multiple subimages
def fix_relative_coord(output_dict, list_of_splits, image_position):
    output_dict_adj = output_dict

    # Getting the image width out of the list of splits (it's the right side of
    # the last split).
    image_width = list_of_splits[-1][1]

    # Getting the split width
    split_width = list_of_splits[image_position][1] - list_of_splits[image_position][0]
    #print("\n\nsplit_width: ")
    #print(split_width)
    #print("\n\n\n")
    #print("list_of_splits[image_position][0]")
    #print(list_of_splits[image_position][0])
    #print("list_of_splits[image_position][1]")
    #print(list_of_splits[image_position][1])

    # First we get a constant adjustment for the "image position". The
    # adjustment is where the left side of the current image starts, relative
    # to the entire image. We can get this from the list_of_splits.
    position_adjustment = list_of_splits[image_position][0] / image_width
    #print("Position adjustment")
    #print(position_adjustment)

    # Now we adjust the x coordinates of the 'detection_boxes' ndarray, We
    # don't need to adjust the y coordinates because we only split on the x. If
    # later I add splitting on y, then the y coordinates need to be adjusted.
    # This adjustment "shrinks" the relative coordinates down.
    adjusted_boxes = output_dict['detection_boxes']
    adjusted_boxes[:,[1,3]] *= (split_width / image_width)

    # Adding the adjustment for which split image it is (the first image
    # doesn't need adjustment, hence the if statement).
    if image_position > 0:
        adjusted_boxes[:,[1,3]] += position_adjustment
        

    # Now adding back in the adjusted boxes to the original ndarray
    output_dict_adj['detection_boxes'] = adjusted_boxes

    return(output_dict_adj)


# Non-max suppression function
def do_non_max_suppression(input_dictionary):
    # The actual nms comes from Tensorflow
    nms_vec = tf.image.non_max_suppression(
        input_dictionary['detection_boxes'],
        input_dictionary['detection_scores'],
        100000,
        iou_threshold=0.5,
        score_threshold=float('-inf'),
        name=None)

    # Converting into a ndarray
    nms_vec_ndarray = tf.Session().run(nms_vec)

    print("\n\n\nthe nms tensor is:")
    print(nms_vec)
    print("the nms ndarray is:")
    print(nms_vec_ndarray)
    print(len(nms_vec_ndarray))
    print("the length of the input array is:")
    print(len(output_dict['detection_boxes']))
    print("\n\n\n")

    # Indexing the input dictionary with the output of non_max_suppression,
    # which is the list of boxes (and score, class) to keep.
    out_dic = input_dictionary.copy()
    out_dic['detection_boxes'] = input_dictionary['detection_boxes'][nms_vec_ndarray].copy() 
    out_dic['detection_scores'] = input_dictionary['detection_scores'][nms_vec_ndarray].copy() 
    out_dic['detection_classes'] = input_dictionary['detection_classes'][nms_vec_ndarray].copy() 

    # Change to output dictionary
    return(out_dic)

# Function for deleting boxes that are unrealistically large.
def delete_giant_boxes(input_dictionary, list_of_splits):
    image_width = list_of_splits[-1][1]
    x_box_max = 300
    rel_x_box_max = x_box_max / image_width
    y_box_max = 400
    rel_y_box_max = y_box_max / image_width
    coord_list = input_dictionary['detection_boxes']
    coord_counter = 0
    delete_list = []

    for coord in coord_list:
        xmin = coord[1]
        xmax = coord[3]
        x = xmax - xmin

        ymin = coord[0]
        ymax = coord[2]
        y = ymax - ymin

        if (x > rel_x_box_max) or (y > rel_y_box_max):
            delete_list.append(coord_counter)
            

        coord_counter += 1

    # Deleting the boxes that are too big
    print("Number of unreasonably large boxes deleted:")
    print(len(delete_list))
    print("\n")
    out_dic = input_dictionary.copy()

    if len(delete_list) > 0:
        print("deleting boxes in original dict")
        out_dic['detection_boxes'] = np.delete(input_dictionary['detection_boxes'], delete_list, 0) 
        out_dic['detection_classes'] = np.delete(input_dictionary['detection_classes'], delete_list, 0) 
        out_dic['detection_scores'] = np.delete(input_dictionary['detection_scores'], delete_list, 0) 

    return(out_dic)


# Setting some stuff up for the totals
image_names = list()
fluorescent_totals = list()
nonfluorescent_totals = list()

# Setting up a list for the detailed output
detailed_results = [["x_min"], ["x_max"], 
    ["y_min"], ["y_max"], 
    ["score"], ["class"], ["name"]]

# Main basically
for image_path in TEST_IMAGE_PATHS:
    # Sets the image position counter for the relative coordinate fix
    image_position_counter = 0

    image = Image.open(image_path)
    image_name_string = str(os.path.splitext(os.path.basename(image_path))[0])
    print('\nprocessing ' + image_name_string + '\n')
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)

    # Here we set up the image splits, based on the users desired number of
    # sub-images, including the overlap. The splits will be a list of sets of
    # two numbers, the lower and upper bounds of the splits.
    splits = get_splits(image_np.shape[1], args.image_split_num, args.overlap_width)
    print(splits)
        
    # Here's where the actual splitting happens
    #split_image_np = np.array_split(image_np, args.image_split_num, axis=1)
    split_image_np = split_image(image_np, splits)
    print("\nThe split image array list looks like:")
    for array in split_image_np:
        print(array.shape)
    
    # Running the inference for the first split, or in the case of a split number
    # of 1, the only split.
    output_dict = run_inference_for_single_image(split_image_np[0], detection_graph)
    
    if args.image_split_num > 1:
        # Getting rid of edge boxes for the first split image
        output_dict = remove_edge_boxes(
            output_dict, 
            splits,
            image_position_counter)

        # Fixing the relative coordinates for the first split image
        output_dict = fix_relative_coord(
            output_dict, 
            splits, 
            image_position_counter)

        image_position_counter = image_position_counter + 1

    # Inference for the following splits, if there's more than 1.
    if args.image_split_num > 1:
        # Fixing the relative coordinates for the first image
        

        # Goes through the image sub-arrays, skipping the first one since we
        # already did that one and the new data will be appended to it.
        for image_split in split_image_np[1:]:
            print("\nProcessing split image. Image position counter:")
            print(str(image_position_counter) + '\n')

            # Running the inference
            split_output_dict = run_inference_for_single_image(image_split, detection_graph)
        
            # Getting rid of edge boxes
            split_output_dict = remove_edge_boxes(
                split_output_dict, 
                splits,
                image_position_counter)
            
            # Correcting the relative coordinates
            split_output_dict = fix_relative_coord(
                split_output_dict, 
                splits, 
                image_position_counter)

            # Adding the new data to the output dict
            output_dict['detection_boxes'] = np.concatenate((
                output_dict['detection_boxes'], 
                split_output_dict['detection_boxes']))
            output_dict['detection_classes'] = np.concatenate((
                output_dict['detection_classes'], 
                split_output_dict['detection_classes']))
            output_dict['detection_scores'] = np.concatenate((
                output_dict['detection_scores'], 
                split_output_dict['detection_scores']))

            #print(output_dict['detection_boxes'])
            #print(output_dict['detection_classes'])
            #print(output_dict['detection_scores'])

            image_position_counter = image_position_counter + 1

    # I'll delete any giant boxes here
    output_dict = delete_giant_boxes(output_dict, splits)

    # Now the I have the output from the sub-images all combined together, I'll
    # do another round of non-maximum suppression to remove the redundant boxes
    # on the edges. This should also fix the problem where the model predicts
    # fluorescent and nonfluorescent for the same seed. It should keep the one
    # with the higher detection score.
    output_dict = do_non_max_suppression(output_dict)

    # Adding in a bit here to count the total number of detections
    seed_counts = get_object_counts(output_dict, args.min_score_threshold)

    # Adding the numbers to the output lists
    image_names.append(image_name_string)
    fluorescent_totals.append(seed_counts[0])
    nonfluorescent_totals.append(seed_counts[1])

    # Getting detailed info about the boxes and scores for ouput (replace this
    # with ndarrays)
    image_detailed_results = get_boxes_and_scores(output_dict, image_name_string)
    detailed_results[0].extend(image_detailed_results[0])
    detailed_results[1].extend(image_detailed_results[1])
    detailed_results[2].extend(image_detailed_results[2])
    detailed_results[3].extend(image_detailed_results[3])
    detailed_results[4].extend(image_detailed_results[4])
    detailed_results[5].extend(image_detailed_results[5])
    detailed_results[6].extend(image_detailed_results[6])

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        line_thickness=2,
        max_boxes_to_draw=10000,
        min_score_thresh=args.min_score_threshold)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imsave(args.output_path + '/' + image_name_string + "_plot" + ".jpg", image_np)

# Printing the summary lists to a file
with open(args.output_path + '/' + 'output.tsv', 'w') as output_file:
    writer = csv.writer(output_file, delimiter='\t')
    writer.writerows(zip(image_names, fluorescent_totals, nonfluorescent_totals))

# Printing the detailed lists to a file
with open(args.output_path + '/' + 'detailed_output.tsv', 'w') as output_file:
    writer = csv.writer(output_file, delimiter='\t')
    writer.writerows(zip(detailed_results[0],
        detailed_results[1],
        detailed_results[2],
        detailed_results[3],
        detailed_results[4],
        detailed_results[5],
        detailed_results[6]))

