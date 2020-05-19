from mrcnn.config import Config

import numpy as np


class HelipadConfig(Config):
    
    """
    Define a configuration for the model
    """
    
    # define the name of the configuration
    NAME = "helipad_cfg_13_no47du_all"
    # number of classes (background + helipad)
    NUM_CLASSES = 1 + 1
    # number of training steps per epoch, ie: number of images/epoch
    STEPS_PER_EPOCH = 4096
    VALIDATION_STEPS = 200
    # number of epochs
    EPOCHS = 1000
    # layers to train
    #               heads: The RPN, classifier and mask heads of the network
    #               all: All the layers
    #               3+: Train Resnet stage 3 and up
    #               4+: Train Resnet stage 4 and up
    #               5+: Train Resnet stage 5 and up
    LAYERS = "all"
    # FOR PREDICTION
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    # # Image mean (RGB)
    MEAN_PIXEL = np.array([105.53481742492863, 108.17376197983675, 95.31055683666851])
    
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 640
    
    
