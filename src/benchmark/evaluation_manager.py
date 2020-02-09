import os

from numpy import expand_dims
from numpy import mean
from numpy import isnan

from tqdm import tqdm

from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from mrcnn.model import MaskRCNN

import sys
sys.path.append('../')

from training.helipad_dataset import HelipadDataset
from prediction_config import PredictionConfig
from training.helipad_config import HelipadConfig

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class EvaluationManager:

    def __init__(self, image_folder, meta_folder, model_folder, weights_filepath=None, include_augmented=False):

        self.image_folder = image_folder
        self.meta_folder = meta_folder
        self.model_folder = model_folder
        self.weights_filepath = weights_filepath

        print("Loading Train Set")
        self.train_set = self.prepare_set(is_train=True, include_augmented=include_augmented)
        print('Train: %d' % len(self.train_set.image_ids))
        print("Loading Test Set")
        self.test_set = self.prepare_set(is_train=False, include_augmented=include_augmented)
        print('Test: %d' % len(self.test_set.image_ids))
        print("Loading Config")
        self.config = HelipadConfig()
        print("Config Loaded")
        print("Loading Model")
        self.model = self.model_predict_setup(weights_filepath)
        print("Model Loaded")

    # Duplicated with training manager
    def prepare_set(self, is_train=True, include_augmented=False):
        set = HelipadDataset()
        set.load_dataset(self.image_folder, self.meta_folder, is_train=is_train, include_augmented=include_augmented)
        set.prepare()
        return set

    # Change to load the model of the last epoch
    # Code duplicated from training manager
    def model_predict_setup(self, predict_weights_filepath=None):
        model_predict = MaskRCNN(mode='inference', model_dir=self.model_folder, config=self.config)

        # find the latest model weight of the training

        # if not predict_weights_filepath:
        #     nb_epoch_done = len(os.listdir(self.model.log_dir))-1
        #     checkpoint_path = os.path.abspath(self.model.checkpoint_path)
        #     folder_name = os.path.dirname(checkpoint_path)
        #     checkpoint_basename = os.path.basename(checkpoint_path)
        #     checkpoint_basename = checkpoint_basename.split('_')
        #     checkpoint_basename[4] = "{:04d}.h5".format(nb_epoch_done)
        #     last_weight_filename = '_'.join(checkpoint_basename)
        #     print(last_weight_filename)
        #     last_weights_filepath = os.path.join(folder_name, last_weight_filename)
        # else:
        #     last_weights_filepath = os.path.join(self.model_folder, predict_weights_filepath)

        model_predict.load_weights(os.path.join(self.model_folder, self.weights_filepath),
                                        by_name=True)

        return model_predict

    def evaluate_model(self, is_train=False):
        if is_train:
            dataset = self.train_set
        else:
            dataset = self.test_set
        APs = list()
        for i in tqdm(range(len(dataset.image_ids))):
            image_id = dataset.image_ids[i]
            try:
                # load image, bounding boxes and masks for the image id
                image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, self.config, image_id,
                                                                                 use_mini_mask=False)
            except:
                print("Image_id {} doesn't exist".format(i))
                image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, self.config, dataset.image_ids[0],
                                                                                 use_mini_mask=False)
            # convert pixel values (e.g. center)
            scaled_image = mold_image(image, self.config)
            # convert image into one sample
            sample = expand_dims(scaled_image, 0)
            # make prediction
            yhat = self.model.detect(sample, verbose=0)
            # extract results for first sample
            r = yhat[0]
            # calculate statistics, including AP
            AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
            if isnan(AP):
                print("AP({}) is nan".format(image_id))
                continue
            # store
            APs.append(AP)
        # calculate the mean AP across all images
        mAP = mean(APs)
        return mAP

    def run(self):

        train_mAP = self.evaluate_model(is_train=True)
        print("Train mAP: %.3f" % train_mAP)
        # evaluate model on test dataset
        test_mAP = self.evaluate_model(is_train=False)
        print("Test mAP: %.3f" % test_mAP)

        w_path = os.path.split(self.weights_filepath)
        output_filename = 'training_results_' + '_'.join(w_path) + '.txt'

        with open(output_filename, 'w') as f:
            f.write("Train mAP: %.3f" % train_mAP)
            f.write("Test mAP: %.3f" % test_mAP)


if __name__ == "__main__":

    # image_folder = os.path.join('C:\\', 'Users', 'jonas', 'Desktop', 'Helipad', 'Helipad_DataBase', 'Helipad_DataBase_original')
    # meta_folder = os.path.join('C:\\', 'Users', 'jonas', 'Desktop', 'Helipad', 'Helipad_DataBase_meta', 'Helipad_DataBase_meta_original')
    # model_folder = os.path.join('C:\\', 'Users', 'jonas', 'Desktop', 'Helipad', 'model')

    image_folder = "../../Helipad_DataBase"
    meta_folder = "../../Helipad_DataBase_meta"
    model_folder = "../../model"

    include_augmented = False

    predict_weights_filepath = 'helipad_cfg_8_no47_aug2_3+20200108T0600/mask_rcnn_helipad_cfg_8_no47_aug2_3+_0472.h5'

    evaluation_manager = EvaluationManager(image_folder,
                                           meta_folder,
                                           model_folder,
                                           predict_weights_filepath,
                                           include_augmented)

    evaluation_manager.run()