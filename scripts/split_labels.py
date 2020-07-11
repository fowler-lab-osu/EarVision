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
split_labels
Cedar Warman

Modified from Dat Tran's split labels notebook:
https://github.com/datitran/raccoon_dataset/blob/master/split%20labels.ipynb

Takes a csv file of annotations (created from pascal voc formatted xmls) and
splits it into two csv files, one for training and one for testing, that can
then be converted to tfrecords.

Usage:

split_labels.py -i <input.csv> -r <train_validation_ratio> -o <output_dir>

"""

import numpy as np
import pandas as pd
import argparse

# Setting up argparse
# parser = argparse.ArgumentParser(description='Splits labels')
# parser.add_argument('-i', 
#                     '--input_path',
#                     type=str,
#                     help=('Input csv file path'))
# parser.add_argument('-r',
#                     '--train_validation_ratio',
#                     type=float,
#                     default=0.7,
#                     help=('Ratio of train to validation'))
# parser.add_argument('-o',
#                     '--output_path',
#                     type=str,
#                     default='.',
#                     help=('Output directory path'))
# args = parser.parse_args()

# Importing the csv input
full_labels = pd.read_csv(args.input_path)

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
train_output_path = args.output_path + '/train_labels.csv'
validation_output_path = args.output_path + '/val_labels.csv'

train.to_csv(train_output_path, index=None)
val.to_csv(validation_output_path, index=None)

