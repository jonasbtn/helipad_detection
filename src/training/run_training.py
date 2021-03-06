import os
import imgaug

from numpy import expand_dims
from numpy import mean
from numpy import isnan

from tqdm import tqdm

import imgaug as ia
from imgaug import augmenters as iaa

from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image

from helipad_detection.src.training.helipad_config import HelipadConfig
from helipad_detection.src.training.helipad_dataset import HelipadDataset
from helipad_detection.src.training.training_manager import TrainingManager

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class RunTraining:
    
    """
    Run a Mask-RCNN training on a dataset. The model has to be configured first in `HelipadConfig`.
    """
    def __init__(self, root_folder, root_meta_folder, model_folder,
                 weights_filename, include_augmented=False,
                 augmented_version=[], predict_weights_filepath=None,
                 train_categories=None, test_categories=None):
        """
        `root_folder`: string, path to the root folder of the dataset containing the original dataset and the different augmented dataset\n
        `root_meta_folder`: string, path to the root folder of the meta dataset containing the original meta dataset and the different augmented meta dataset\n
        `model_folder`: the folder where to store/load the model weights\n
        `weights_filename`: the initial model weights filename from where to start the training. The script supposed that the full path of the weights is `os.path.join(model_folder, weights_filename)`\n
        `include_augmented`: boolean, True if the dataset has to include augmented images\n
        `augmented_version`: list of integer, specifying the augmented dataset version to include in the dataset\n
        `predict_weights_filepath`: the model weights filename to use when computing the mAP of the model after the training. If None, the weights obtained at the last epoch is used.\n
        `train_categories`:a list of string, specifying the categories to be included in the training set. If None, all the categories are included.\n
        `test_categories`:a list of string, specifying the categories to be included in the test set. If None, all the categories are included.\n
        """

        self.training_manager = TrainingManager(root_folder,
                                                root_meta_folder,
                                                model_folder,
                                                weights_filename,
                                                include_augmented,
                                                augmented_version,
                                                train_categories,
                                                test_categories)

        self.training_manager.model_setup()

        self.predict_weights_filepath = predict_weights_filepath

    def set_augmentation(self):
        augmentation = imgaug.augmenters.Sometimes(0.5, [
                            imgaug.augmenters.Fliplr(0.5),
                            imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                        ])
#         policy = iaa.Sequential([
#                             iaa.Sometimes(0.2, iaa.Fliplr(1)),
#                             iaa.Sometimes(0.2, iaa.Flipud(1)),
#                             iaa.Sometimes(0.2, iaa.Affine(rotate=(-45, 45))),
#                             iaa.Sometimes(0.2, iaa.Affine(rotate=(-90, 90))),
#                             iaa.Sometimes(0.2, iaa.Affine(scale=(0.5, 1.5))),
#                             iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0.0, 3.0))),
#                             iaa.Sometimes(0.1, iaa.AllChannelsHistogramEqualization()),
#                             iaa.Sometimes(0.2, iaa.ShearX((-20, 20))),
#                             iaa.Sometimes(0.2, iaa.ShearY((-20, 20)))
#                             ])
        return augmentation

    def evaluate_model(self, is_train=False):
        """
        Evaluate the model by computing the mAP on the training/test, depending if `is_train` is true of false.\n
        Returns: a float representing the mAP
        """
        model = self.training_manager.model_predict
        config = self.training_manager.config
        if is_train:
            dataset = self.training_manager.train_set
        else:
            dataset = self.training_manager.test_set
        APs = list()
        for i in tqdm(range(len(dataset.image_ids))):
            image_id = dataset.image_ids[i]
            image_path = dataset.source_image_link(image_id)
            elements = os.path.basename(image_path).split('_')
            if elements[-2] == 'augmented':
                continue
            try:
                # load image, bounding boxes and masks for the image id
                image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, config, image_id,
                                                                     use_mini_mask=False)
            except:
                print("Image_id {} doesn't exist".format(i))
                image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, config, dataset.image_ids[0],
                                                                                 use_mini_mask=False)

            # convert pixel values (e.g. center)
            scaled_image = mold_image(image, config)
            # convert image into one sample
            sample = expand_dims(scaled_image, 0)
            # make prediction
            yhat = model.detect(sample, verbose=0)
            # extract results for first sample
            r = yhat[0]
            # calculate statistics, including AP
            AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
            # store
            if isnan(AP):
                print("AP({}) is nan".format(image_id))
                continue
            APs.append(AP)
        # calculate the mean AP across all images
        mAP = mean(APs)
        return mAP

    def run(self):
        """
        Run the training
        """
        # train weights (output layers or 'heads')
        self.training_manager.model.train(self.training_manager.train_set,
                                          self.training_manager.test_set,
                                          learning_rate=self.training_manager.config.LEARNING_RATE,
                                          epochs=self.training_manager.config.EPOCHS,
                                          layers=self.training_manager.config.LAYERS,
                                          augmentation=None)

    def run_predict(self):
        """
        Run the model evaluation by computing the mAP on the train set and on the test test.\n
        Print the results and writes them in a file named after the weights filename. 
        """
        self.training_manager.model_predict_setup(self.predict_weights_filepath)

        train_mAP = self.evaluate_model(is_train=True)
        print("Train mAP: %.3f" % train_mAP)
        # evaluate model on test dataset
        test_mAP = self.evaluate_model(is_train=False)
        print("Test mAP: %.3f" % test_mAP)

        w_path = os.path.split(self.predict_weights_filepath)
        output_filename = 'training_results_' + '_'.join(w_path) + '.txt'

        with open(output_filename, 'w') as f:
            f.write("Train mAP: %.3f" % train_mAP)
            f.write("Test mAP: %.3f" % test_mAP)


if __name__ == "__main__":

    # root_folder = os.path.join('C:\\', 'Users', 'jonas', 'Desktop', 'Helipad', 'Helipad_DataBase')
    # root_meta_folder = os.path.join('C:\\', 'Users', 'jonas', 'Desktop', 'Helipad', 'Helipad_DataBase_meta')
    # model_folder = os.path.join('C:\\', 'Users', 'jonas', 'Desktop', 'Helipad', 'model')

#     root_folder = "../../../Helipad/Helipad_DataBase"
#     root_meta_folder = "../../../Helipad/Helipad_DataBase_meta"
    # model_folder = "../../../Helipad/model"
    
    root_folder = "D:\\Jonas\\Helipad_DataBase"
    root_meta_folder = "D:\\Jonas\\Helipad_DataBase_meta"
    model_folder = "D:\\Jonas\\model\\"
    include_augmented = True
    augmented_version = [11]

    train_categories = ["1", "2", "3", "5", "6", "8", "9"]
    test_categories = ["1", "2", "3", "5", "6", "8", "9"]

    # weights_filename = 'helipad_cfg_10_no47du_all20200420T0127/mask_rcnn_helipad_cfg_10_no47du_all_0896.h5'
    # weights_filename = 'helipad_cfg_11_no47du_3+20200507T2024/mask_rcnn_helipad_cfg_11_no47du_3+_0038.h5'
    # weights_filename = 'helipad_cfg_12_no47du_all20200513T1024/mask_rcnn_helipad_cfg_12_no47du_all_0113.h5'
    weights_filename = 'helipad_cfg_13_no47du_all20200514T2105/mask_rcnn_helipad_cfg_13_no47du_all_0057.h5'
    base_weights = 'mask_rcnn_coco.h5'

    predict_weights_filepath = 'helipad_cfg_7_aug123_all20200106T2012/mask_rcnn_helipad_cfg_7_aug123_all_0472.h5'

    run_training = RunTraining(root_folder,
                               root_meta_folder,
                               model_folder,
                               weights_filename,
                               include_augmented=include_augmented,
                               augmented_version=augmented_version,
                               predict_weights_filepath=None,
                               train_categories=train_categories,
                               test_categories=test_categories)

    print('Starting Training')
    run_training.run()
    print('Training Over')

    print('Evaluating Last Epoch')
    run_training.run_predict()
    print('Evaluation Done')
