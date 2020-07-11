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


# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Binary to run train and evaluation on object detection model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import tensorflow.compat.v1 as tf
import os
from pathlib import Path
import argparse
import sys
import pyfiglet

"""
Setting up argument passing
"""

parser = argparse.ArgumentParser(description='Seed Project Training Tool')
parser.add_argument('-o',
                    '--od_dir',
                    type=str,
                    help=('Path to object detection directory'),
                    default='./seed_models/research/')
parser.add_argument('-p',
                    '--pipeline_dir',
                    type=str,
                    help=('Path to train config file'),
                    default='./utils/training/train.config')
parser.add_argument('-m',
                    '--model_dir',
                    type=str,
                    help=('Path to model'),
                    default='./utils/training/models/model')
parser.add_argument('-n',
                    '--num_train_steps',
                    type=int,
                    help=('Number of training steps to be used'),
                    required=True)
parser.add_argument('-s',
                    '--one_n_eval_examples',
                    type=int,
                    default=3,
                    help=('Number of 1 of N evaluation examples'))
args = parser.parse_args()

""" 
Setting up environmental variables for object detection model
"""

# handle / on input path
newPath = args.od_dir

if newPath[0] == '/':
    newPath = newPath[1:]

netsPath = newPath + "/tf_slim"


# create new path
sys.path.append(newPath) 
sys.path.append(netsPath)


from object_detection import model_hparams
from object_detection import model_lib

def main(unused_argv):
  """
  Pretty banner 
  """

  ascii_banner = pyfiglet.figlet_format("Seed Project Training Tool")
  print(ascii_banner)

  config = tf.estimator.RunConfig(model_dir=args.model_dir)

  train_and_eval_dict = model_lib.create_estimator_and_inputs(
      run_config=config,
      hparams=model_hparams.create_hparams(None),
      pipeline_config_path=args.pipeline_dir,
      train_steps=args.num_train_steps,
      sample_1_of_n_eval_examples=args.one_n_eval_examples,
      sample_1_of_n_eval_on_train_examples=(
          args.one_n_eval_examples))
  estimator = train_and_eval_dict['estimator']
  train_input_fn = train_and_eval_dict['train_input_fn']
  eval_input_fns = train_and_eval_dict['eval_input_fns']
  eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
  predict_input_fn = train_and_eval_dict['predict_input_fn']
  train_steps = train_and_eval_dict['train_steps']

  train_spec, eval_specs = model_lib.create_train_and_eval_specs(
        train_input_fn,
        eval_input_fns,
        eval_on_train_input_fn,
        predict_input_fn,
        train_steps,
        eval_on_train_data=False)

  # Currently only a single Eval Spec is allowed.
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])

if __name__ == '__main__':
  tf.app.run()
