from mrcnn.config import Config


class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "helipad_cfg"
	# number of classes (background + kangaroo)
	NUM_CLASSES = 1 + 1
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
