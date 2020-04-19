from mrcnn.config import Config


# define a configuration for the model
class HelipadConfig(Config):
    # define the name of the configuration
    NAME = "helipad_cfg_10_no47du_3+"
    # number of classes (background + helipad)
    NUM_CLASSES = 1 + 1
    # number of training steps per epoch, ie: number of images/epoch
    STEPS_PER_EPOCH = 256
    # number of epochs
    EPOCHS = 700
    # layers to train
    #               heads: The RPN, classifier and mask heads of the network
    #               all: All the layers
    #               3+: Train Resnet stage 3 and up
    #               4+: Train Resnet stage 4 and up
    #               5+: Train Resnet stage 5 and up
    LAYERS = "3+"
    # FOR PREDICTION
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
