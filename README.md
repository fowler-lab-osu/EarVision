# EarVision

Maize fluorescent kernel object detection in TensorFlow 1.14.0

## Setup:

- Clone the seed project's version of TensorFlow's model zoo located at [this Github repo](https://github.com/fowler-lab-osu/EarVision_TensorFlow_Object_Detection_API) into the base directory 

- Set up an Anaconda environment using the `seed_conda.yml` file provided in this repo. Activate this environment

- Prepare a directory containing the `.png` files you would like to train or run inference on 

- Run the data preperation tool followed by whichever tools are approriate 

## Data Preparation 

To prepare `.png` images for use with this model, one can use the provided data preperation tool. Running the tool as described below will take your input images and create an output directory containing cropped versions of your images, `train_labels.csv`, `val_labels.csv`, `train.record`, `val.record`, and an annotations directory all while leaving your original data untouched.

### Usage

Example command:
`python data_prep.py -i data/training/2019/`

##### Options:
    -i : (required) path to data input directory (ie .png images)
    -o : path to output directory (default is ./output/data)
    -m : path to model directory (default is ./seed_models/research/)
    -n : ratio of training to validation data (default is 0.7)
    -q : if True, enables quiet mode which reduces amount of terminal output (default is False)

*Note:* If you use an output path different from the default one, you will have to change the paths in `utils/training/train.config` to reflect this before moving on to the next tools

## Training

The training tool will train on the sample . It uses transfer learning from a Faster RCNN Inception Resnet V2 model trained on COCO (from 1/28/2018). 

### Usage

Example command: `python train.py -n 50000`

##### Options:
    -o : path to object detection directory (default is ./seed_models/research/)
    -p : path to training config file (default is ./utils/training/train.config)
    -m : path to model directory (default is ./utils/training/models/model)
    -n : (required) number of training steps to be used
    -s : number of 1 of N evalutation examples (default is 3)

## Inference

The inference tool will use the checkpoints created by the training tool to perform inferences on provided input images. Output is given in the form of new .jpg images which feature labels for the different corn kernels detected. The default location for the output images will be in `./output/inference/` unless otherwise specified by the user.

### Usage

*Note:* If you want to use the checkpoints you generated from the training tool for inference, you must go to the model directory specified for the training tool (-m option) and find the last generated checkpoint file. Then, set `-c` for the inference tool as `../path/to/checkpoint/model.ckpt-#` where `#` is the highest checkpoint number generated. If you don't specify a checkpoint, an included checkpoint (`ckpt-50000`) will be used instead.

Example command: `python inference.py -d ./data/testing/2019 -n 3 -s 0.12`

##### Options:
    -d : (required) path to test image directory 
    -c : path to the checkpoint to be used for inference (default is ./data/default_checks/model.ckpt-50000)
    -l : path to class label map (default is ./utils/training/data/label_map.pbtxt)
    -o : output path (default is ./output/inference/)
    -m : path to model directory (default is ./seed_models/research/)
    -s : minimum score threshold for plotting bounding boxes (default is 0.05)
    -n : number of image subdivisions to run the object detection on (default is 1)
    -w : pixel overlap width for image subdivisions (default is 100)

