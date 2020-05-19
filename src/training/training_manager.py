import os
import argparse

from helipad_detection.src.training.helipad_dataset import HelipadDataset
from helipad_detection.src.training.helipad_config import HelipadConfig

from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from mrcnn.model import MaskRCNN


class TrainingManager:
    
    """
    Manager to setup the training.
    """
    
    def __init__(self, root_folder, root_meta_folder, model_folder,
                 weights_filename, include_augmented=False, augmented_versions=[],
                 train_categories=None, test_categories=None):

        self.root_folder = root_folder
        self.root_meta_folder = root_meta_folder
        self.model_folder = model_folder
        self.weight_filename = weights_filename
        self.include_augmented = include_augmented
        self.augmented_version = augmented_versions

        print("Loading Train Set")
        self.train_set = self.prepare_set(is_train=True, include_augmented=include_augmented, include_categories=train_categories)
        print('Train: %d' % len(self.train_set.image_ids))
        print("Loading Test Set")
        self.test_set = self.prepare_set(is_train=False, include_augmented=False, include_categories=test_categories)
        print('Test: %d' % len(self.test_set.image_ids))
        print("Loading Config")
        self.config = HelipadConfig()

    def prepare_set(self, is_train=True, include_augmented=False, include_categories=None):
        set = HelipadDataset()
        set.load_dataset(self.root_folder, self.root_meta_folder, is_train=is_train,
                         include_augmented=include_augmented, augmented_versions=self.augmented_version,
                         include_categories=include_categories)
        set.prepare()
        return set

    def display_samples(self):
        for i in range(20):
            # define image id
            image_id = i
            # load the image
            image = self.train_set.load_image(image_id)
            # load the masks and the class ids
            mask, class_ids = self.train_set.load_mask(image_id)
            # extract bounding boxes from the masks
            bbox = extract_bboxes(mask)
            # display image with masks and bounding boxes
            display_instances(image, bbox, mask, class_ids, self.train_set.class_names)

    def model_setup(self):
        # define the model
        self.model = MaskRCNN(mode='training', model_dir=self.model_folder, config=self.config)
        # load weights (mscoco) and exclude the output layers
        self.model.load_weights(os.path.join(self.model_folder, self.weight_filename),
                               by_name=True,
                               exclude=["mrcnn_class_logits",
                                        "mrcnn_bbox_fc",
                                        "mrcnn_bbox",
                                        "mrcnn_mask"])

    # Change to load the model of the last epoch
    def model_predict_setup(self, predict_weights_filepath=None):
        self.model_predict = MaskRCNN(mode='inference', model_dir=self.model_folder, config=self.config)
        # find the latest model weight of the training

        if not predict_weights_filepath:
            nb_epoch_done = len(os.listdir(self.model.log_dir))-1
            checkpoint_path = os.path.abspath(self.model.checkpoint_path)
            folder_name = os.path.dirname(checkpoint_path)
            checkpoint_basename = os.path.basename(checkpoint_path)
            checkpoint_basename = checkpoint_basename.split('_')
            checkpoint_basename[4] = "{:04d}.h5".format(nb_epoch_done)
            last_weight_filename = '_'.join(checkpoint_basename)
            print(last_weight_filename)
            last_weights_filepath = os.path.join(folder_name, last_weight_filename)
        else:
            last_weights_filepath = os.path.join(self.model_folder,predict_weights_filepath)
        self.model_predict.load_weights(last_weights_filepath,
                                        by_name=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='display_sample', default=False)
    parser.add_argument('-c', dest='display_config', default=True)
    args = parser.parse_args()

    display_sample = args.display_sample
    display_config = args.display_config

    root_folder = os.path.join('C:\\', 'Users', 'jonas', 'Desktop', 'Helipad', 'Helipad_DataBase')
    root_meta_folder = os.path.join('C:\\', 'Users', 'jonas', 'Desktop', 'Helipad', 'Helipad_DataBase_meta')
    model_folder = os.path.join('C:\\', 'Users', 'jonas', 'Desktop', 'Helipad', 'model')

    # image_folder = "../Helipad_DataBase"
    # meta_folder = "../Helipad_DataBase_meta"
    # model_folder = "../model"

    weights_filename = 'mask_rcnn_coco.h5'

    include_augmented = True
    augmented_version = [1]

    training_manager = TrainingManager(root_folder,
                                       root_meta_folder,
                                       model_folder,
                                       weights_filename,
                                       include_augmented,
                                       augmented_version)

    if display_config:
        training_manager.config.display()

    if display_sample:
        training_manager.display_samples()
