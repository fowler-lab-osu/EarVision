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
crop_annotations_and_images.py
Cedar Warman

This script takes in images and annotations in PASCAL VOC format and crops them
into smaller images and assocaited annotations. This is to be used as training
data when memory is limiting, which is often the case for maize ear annotated
images with 300-600 bounding boxes per image.

Usage:
crop_annotations_and_images.py 
    -i <input_dir> 
    -o <output_dir>
    -n <image_split_number>

"""

import os
import io
import cv2
from copy import deepcopy
from pathlib import Path
import numpy as np
import argparse
from lxml import etree
from PIL import Image

# Setting up arguments
# parser = argparse.ArgumentParser(description='Crop annotations and imagmes')
# parser.add_argument('-i',
#                     '--input_dir',
#                     type=str,
#                     help=('Path to input directory'))
# parser.add_argument('-o',
#                     '--output_dir',
#                     type=str,
#                     help=('Path to output directory'))
# parser.add_argument('-n',
#                     '--image_split_num',
#                     type=int,
#                     default=3,
#                     help=('Number of subdivisions to crop into.'))
# args = parser.parse_args()


"""
============================
load_image_into_numpy_array
============================
"""
def load_image_into_numpy_array(image):
    imported_image = cv2.imread(image)
    return(imported_image)


"""
====================
split_image_np_array
====================
"""
def split_image_np_array(image, image_split_num):
    split_image = np.array_split(image, image_split_num, axis=1)
    return(split_image)


"""
=================
save_split_images
=================
"""
def save_split_images(split_image, output_dir, image_name):
    counter = 1
    for image in split_image:
        output_string = output_dir + "/" + image_name + "_s" + str(counter) + ".png"
        cv2.imwrite(output_string, image)
        counter += 1
    


"""
=================
split_annotations
=================
"""
def split_annotations (xml_path, split_number, image_dimensions_list):
    # Importing the xml
    tree = etree.parse(xml_path) 

    # Getting the name of the xml file
    xml_basename = tree.xpath('filename')[0].text[:-4]

    # Getting the dimensions for this image, set of x dimensions based on the
    # split number.
    dimensions = image_dimensions_list.get(xml_basename)

    #print("dimensions are ")
    #print(dimensions)

    # Grabbing the width of the image and making a variable for the current min
    # and max values of the bins.
    #image_width = int(tree.xpath('//width')[0].text)
    #bin_width = image_width // split_number 
    current_bin_min = 0
    current_bin_max = dimensions[0]

    # Setting up empty lists to contain the split xml files
    tree_list = []

    # Copying the tree to the list (each list entry will be the annotations or
    # one sub-image). I need to use deepcopy because all the copies point to
    # the same data, so if any single copy is edited (aka an element is
    # deleted) then all the copies are edited. 
    for x in range(0, split_number):
        tree_to_add = deepcopy(tree)
        tree_list.append(tree_to_add)

    # Going through the split trees one by one.
    # TODO Make a counter and some "acceptable" ymin and ymax values the the
    # bounding boxes have to fall into or else they will get deleted, or
    # changed in size if they're over the line. The values will then get
    # something added to them when it does the next tree.

    counter = 0
    image_name_counter = 1
    bin_counter = 1
    xml_width_counter = 0

    for split_tree in tree_list:
        #print("\n\n\nProcessing tree\n")

        #print("Current bin min: " + str(current_bin_min))
        #print("Current bin max: " + str(current_bin_max))

        # Min box size (boxes smaller than this width get deleted)
        min_box_size = 30

        # Fixing the xml filename
        new_filename = xml_basename + "_s" + str(image_name_counter) + ".png"
        split_tree.xpath('//filename')[0].text = new_filename

        # Fixing the xml width
        split_tree.xpath('//width')[0].text = str(dimensions[xml_width_counter])

        # Grabbing the objects from the xml
        object_list = split_tree.xpath('//object')
    
        # Looks at each "object" (aka bounding box, which has the format:
        #   <object>
        #       <name>nonfluorescent</name>
        #       <pose>Unspecified</pose>
        #       <truncated>1</truncated>
        #       <difficult>0</difficult>
        #       <bndbox>
        #           <xmin>208</xmin>
        #           <ymin>693</ymin>
        #           <xmax>265</xmax>
        #           <ymax>746</ymax>
        #       </bndbox>
        #   </object> 
        
        for entry in range(0, len(object_list)):
            current_object = object_list[entry]
            
            xmax = int(float(current_object.xpath('bndbox/xmax')[0].text))
            xmin = int(float(current_object.xpath('bndbox/xmin')[0].text))

            # If the object is totally outside the current bin, it will get
            # deleted here.
            if (xmax <= current_bin_min) or (xmin >= current_bin_max):
                
                # Removing the entire object from the original tree 
                current_object.getparent().remove(current_object)
        
                # This is just for a print I think, delete when finished
                counter += 1

            # If the object crosses the left side of the bin, it will get
            # adjusted here to be fully inside the bin (by cropping it).
            elif (xmax > current_bin_min) and (xmin <= current_bin_min):

                adjusted_xmax = xmax - current_bin_min
                adjusted_xmin = current_bin_min - current_bin_min

                # Checking for tiny boxes. If they're too small they get
                # removed
                if (adjusted_xmax - adjusted_xmin) < min_box_size:
                    current_object.getparent().remove(current_object)

                else:
                    current_object.xpath('bndbox/xmax')[0].text = str(adjusted_xmax)
                    current_object.xpath('bndbox/xmin')[0].text = str(adjusted_xmin)

            # Same as previous, but with the right side of the bin
            elif (xmax >= current_bin_max) and (xmin < current_bin_max):

                adjusted_xmax = current_bin_max - current_bin_min
                adjusted_xmin = xmin - current_bin_min
                
                # Checking for tiny boxes. If they're too small they get
                # removed
                if (adjusted_xmax - adjusted_xmin) < min_box_size:
                    current_object.getparent().remove(current_object)
                
                else:
                    current_object.xpath('bndbox/xmax')[0].text = str(adjusted_xmax)
                    current_object.xpath('bndbox/xmin')[0].text = str(adjusted_xmin)

            # Finally, for the boxes that are totally inside the bin, just
            # fixing the relative coordinates
            elif (xmax < current_bin_max) and (xmin > current_bin_min):
                adjusted_xmax = xmax - current_bin_min
                adjusted_xmin = xmin - current_bin_min

                current_object.xpath('bndbox/xmax')[0].text = str(adjusted_xmax)
                current_object.xpath('bndbox/xmin')[0].text = str(adjusted_xmin)
            
        current_bin_min = current_bin_min + dimensions[bin_counter] 
        current_bin_max = current_bin_max + dimensions[bin_counter]

        #print("Total deleted boxes: " + str(counter))
        counter = 0
        image_name_counter += 1
        xml_width_counter += 1

    return(tree_list)
        

"""
====
Main
====
"""

# def main():
#     # Making the output directories
#     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
#     Path(args.output_dir + "/annotations").mkdir(parents=True, exist_ok=True)
    
#     # Setting up some paths
#     input_dir = Path(args.input_dir)
#     annotations_dir = input_dir / "annotations"

#     # Setting up a dictionary to store the final cropped dimensions (the reason
#     # I did it like this is that the final dimensions come from the image
#     # cropping for loop, but I want to store them in a place that can be pulled
#     # from the xml loop. Not sure if this is the most efficient way, but it
#     # works). For now I'll just store the x, because the y should all be the
#     # same.
#     image_dimensions = {}

#     # Cropping the images
#     for image_file in os.listdir(input_dir):
#         image_path = str(input_dir / image_file)
#         if image_path.endswith(".png"):
#             print("Processing image file: " + str(image_file) + '\n')
#             current_image = str(image_file[:-4])

#             # Importing and splitting up the image
#             image = load_image_into_numpy_array(image_path)
#             split_images = split_image_np_array(image, args.image_split_num)

#             # Saving the images
#             save_split_images(split_images, args.output_dir, current_image)

#             # Getting the exact dimensions of the split image (they might vary
#             # a littlebased on rounding). This will be used on the xml files to
#             # make the dimensions elements reflect the cropped image.
#             x_coord_list = []
#             for sub_image in split_images:
#                 x_coord_list.append(sub_image.shape[1])
#             image_dimensions[current_image] = x_coord_list
#             #print(image_dimensions)

#     # Going through each annotation file
#     for xml_file in os.listdir(annotations_dir):
#         print("Processing xml file: " + str(xml_file) + '\n')
#         xml_path = str(annotations_dir / xml_file)
#         output_tree = split_annotations(xml_path, 3, image_dimensions) 
        
#         # Printing out the subtrees
#         tree_number_print = 1
#         for output in output_tree:
#             tree_name_output_string = (args.output_dir + "/annotations/" + 
#                 xml_file[:-4] + "_s" + str(tree_number_print) + ".xml")

#             xml_string = etree.tostring(output, pretty_print=True)

#             with open(tree_name_output_string, 'wb') as f:
#                 f.write(xml_string)

#             tree_number_print += 1


if __name__ == "__main__":
    main()
