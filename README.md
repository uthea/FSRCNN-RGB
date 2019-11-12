# FSRCNN-RGB
<img src="http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN/img/framework.png">

## Table of Contents
+ [About](#about)
+ [Prerequisite](#prerequisites)
+ [Usage](#usage)

## About <a name = "about"></a>
FSRCNN is a CNN Architecure used to solve image super resolution problem which is faster and lighter to use than the SRCNN.

FSRCNN can also upscale an image for 2x and 3x and still get good result compared to bicubic interpolation.

## Prerequisites <a name = "prerequisites"></a>
+ pytorch and torchvision
+ PIL
+ numpy
+ tensorboard (for logging)

## Usage <a name="usage"></a>
### Creating dataset
Use create_dataset function in data_processing.py to create dataset from raw image.

Parameters for create_dataset function:
+ image_dir = the path folder that contains images to be processed
+ output_dir = path for hd5 file 
+ input_size = size for the input image
+ label_size = size for ground truth image
+ stride = stride for cropping subimage

## Instatiating FSRCNN object
Before training or processing an image, create an object of FSRCNN class.

Parameters of the constructor
+ d, s, c, and m refers to the model hyperparameters (Read this <a href="https://arxiv.org/abs/1608.00367">paper</a>)
+ upscale_factor = stride parameter for ConvolutionalTranspose (deconv layer)
+ num_epoch = number of epoch for a session
+ batch_size = number of data per batch
+ layers_lr = learning rate for all layers except deconv layer
+ deconv_lr = learning rate for deconv layer
+ ckpt = model checkpoint path
+ ckpt_mode = the possible values are 'load' , 'resume' and 'new'
+ padding = accept a list that consist of 5 numbers for layers padding paramameters, the order is [feature_extraction, shrink, lm, expanding, deconv]


## Training the model
Run the train method of FSRCNN object to train the model

Parameters for train method
+ train_deconv_only = freeze all layers except the last layer
+ train_path = path for the h5 training data
+ validation_path = path for the h5 validation data
+ summary_path = path for storing tensorboard event file


## Generating upscaled image
Run upscaled method of FSRCNN object, it only require the path of input image.
Upscaled output is stored in the current directory with file name "output.png"
