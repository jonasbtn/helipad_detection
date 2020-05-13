import os
import cv2
import json
from random import randint
from numpy import expand_dims
import matplotlib.pyplot as plt
from tqdm import tqdm

from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image

import sys
sys.path.append('../')

from training.helipad_config import HelipadConfig
from training.helipad_dataset import HelipadDataset
from training.filter_manager import FilterManager

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class RunDetection:

    def __init__(self, image_folder, output_meta_folder, model_folder, weight_filename, model_number,
                 activate_filter=False, test_only=False):

        self.image_folder = image_folder
        self.output_meta_folder = output_meta_folder
        if not os.path.isdir(output_meta_folder):
            os.mkdir(output_meta_folder)
        self.model_folder = model_folder
        self.weight_filename = weight_filename
        self.model_number = model_number
        self.config = HelipadConfig()
        self.activate_filter = activate_filter
        self.model_predict_setup()

        self.target_files = self.build_target_files(self.image_folder,
                                                    self.output_meta_folder,
                                                    test_only=test_only)
        print("{} files to predict!".format(len(self.target_files)))

    # Change to load the model of the last epoch
    def model_predict_setup(self):
        self.model_predict = MaskRCNN(mode='inference', model_dir=self.model_folder, config=self.config)
        self.model_predict.load_weights(os.path.join(self.model_folder, self.weight_filename),
                                        by_name=True)

    @staticmethod
    def build_target_files(image_folder, meta_folder, test_only=False,
                           tms_dataset=False, zoom_level=None,
                           include_category=None,
                           include_negative=True,
                           city_lat=None):
        target_files = []
        for subdir, dirs, files in os.walk(image_folder, topdown=True):
            for file in files:
                imagepath = os.path.join(subdir, file)
                try:
                    image = cv2.imread(imagepath)
                except:
                    print("File {} does not exist".format(imagepath))
                    continue
                if not tms_dataset:
                    image_name = os.path.splitext(file)[0]
                    image_number = int(image_name.split('_')[1])
                    if test_only and image_number <= 4250:
                        continue
                    meta_filepath = os.path.join(meta_folder,
                                                 os.path.basename(subdir),
                                                 os.path.splitext(file)[0] + ".meta")
                    if not os.path.isfile(meta_filepath):
                        continue
                    if include_category:
                        with open(meta_filepath, 'r') as f:
                            meta = json.load(f)
                        if "groundtruth" not in meta:
                            continue
                        if meta["groundtruth"]["helipad"]:
                            if "category" not in meta["groundtruth"]:
                                continue
                            elif meta["groundtruth"]["category"] not in include_category:
                                continue
                           # this adds the false samples and only the positive samples 
                           # from include_category
                        if not include_negative and not meta["groundtruth"]["helipad"]:
                            continue
                else:
                    dir_zoom_level = os.path.basename(os.path.dirname(subdir))
                    if zoom_level and dir_zoom_level != str(zoom_level):
                        continue
                    xtile = os.path.basename(subdir)
                    print(xtile)
                    print(xtile[:2])
                    if city_lat:
                        if city_lat[1] != xtile[:2]:
                            continue
                    ytile = os.path.splitext(file)[0]
                    meta_filepath = os.path.join(meta_folder,
                                                 dir_zoom_level,
                                                 xtile,
                                                 "Satellite_{}_{}_{}.meta".format(zoom_level,
                                                                                  xtile,
                                                                                  ytile))

                    if not os.path.isfile(meta_filepath):
                        print('meta not found')
                        print(meta_filepath)
                        continue

                target_files.append([imagepath, meta_filepath])
        return target_files

    def run(self):

        for i in tqdm(range(len(self.target_files))):
            image_meta_path = self.target_files[i]

            imagepath = image_meta_path[0]
            meta_filepath = image_meta_path[1]

            image = cv2.imread(imagepath)

            scaled_image = mold_image(image, self.config)
            sample = expand_dims(scaled_image, 0)
            yhat = self.model_predict.detect(sample, verbose=0)

            rois = yhat[0]['rois']
            class_id = yhat[0]['class_ids']
            score = yhat[0]['scores']

            # reorder rois :
            # x1, y1, x2, y2
            bboxes = []
            for roi in rois:
                box = [int(roi[1]), int(roi[0]), int(roi[3]), int(roi[2])]
                bboxes.append(box)

            class_ids = []
            for id in class_id:
                class_ids.append(int(id))

            scores = []
            for s in score:
                scores.append(float(s))

            # filter is helipad detected
            if self.activate_filter and len(scores) > 0:
                # remove if score < threshold (see FilterManager for default values)
                bboxes, class_ids, scores = FilterManager.filter_by_scores(bboxes, class_ids, scores)
                # Filter overlapping box (see FilterManager for default value of threshold_iou and threshold_area)
                bboxes, class_ids, scores = FilterManager.filter_by_iou(bboxes, class_ids, scores)

            if os.path.isfile(meta_filepath):
                with open(meta_filepath, 'r') as f:
                    meta = json.load(f)
            else:
                meta = {}

            if "predicted" in meta:
                predicted = meta["predicted"]
            else:
                predicted = {}

            key = "model_{}".format(self.model_number)

            # Save to meta roi
            predicted[key] = {}
            predicted[key]["box"] = bboxes
            predicted[key]["class_id"] = class_ids
            predicted[key]["score"] = scores

            if len(bboxes) > 0:
                predicted[key]["helipad"] = True
            else:
                predicted[key]["helipad"] = False

            meta["predicted"] = predicted

            with open(meta_filepath, 'w') as f:
                json.dump(meta, f, indent=4, sort_keys=True)

    @staticmethod
    def review_prediction(image_folder, meta_folder, model_number, test_only=False):

        # colors = [(randint(0,255), randint(0,255), randint(0,255))]
        # color = (randint(0,255), randint(0,255), randint(0,255))

        groundtruth_color = (0, 0, 255)
        predict_color = (255, 0, 0)

        for subdir, dirs, files in os.walk(image_folder, topdown=True):
            for file in files:

                imagepath = os.path.join(subdir, file)
                try:
                    image = cv2.imread(imagepath)
                except:
                    print("File {} does not exist".format(imagepath))
                    continue
                image_name = os.path.splitext(file)[0]
                image_number = int(image_name.split('_')[1])
                if test_only and image_number <= 4250:
                    continue
                meta_filepath = os.path.join(meta_folder,
                                             os.path.basename(subdir),
                                             os.path.splitext(file)[0] + ".meta")

                if not os.path.isfile(meta_filepath):
                    print("File {} does not exist".format(meta_filepath))
                    continue

                with open(meta_filepath, 'r') as f:
                    meta = json.load(f)

                if not "predicted" in meta:
                    print("Image {} not predicted".format(file))
                    continue

                predicted = meta["predicted"]

                key = "model_{}".format(model_number)

                model_prediction = predicted[key]

                # if "box" not in model_prediction:
                #     print("File {} has no box".format(file))
                #     continue

                bboxes = model_prediction["box"]
                scores = model_prediction["score"]

                for i in range(len(bboxes)):
                    box = bboxes[i]
                    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), predict_color, 2)
                    cv2.putText(image, "{}:{}".format(model_number, str(scores[i])),
                                (box[0]+10, box[1]),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                predict_color,
                                2,
                                cv2.LINE_AA)

                if "groundtruth" not in meta:
                    print("File {} has no groundtruth".format(file))
                else:
                    groundtruth = meta["groundtruth"]
                    if groundtruth["helipad"]:
                        bboxes = groundtruth["box"]
                        for i in range(len(bboxes)):
                            box = bboxes[i]
                            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), groundtruth_color, 2)

                # colors.append((randint(0,255), randint(0,255), randint(0,255)))

                # key = "predicted_" + str(j)
                # print(key)
                # print("here")

                cv2.imshow('image', image)
                k = cv2.waitKey(0)

                # plt.imshow(image)
                # plt.show()


if __name__ == "__main__":

    # image_folder = "C:\\Users\\jonas\\Desktop\\Helipad\\Helipad_DataBase\\Helipad_DataBase_original"
    # meta_folder = "C:\\Users\\jonas\\Desktop\\Helipad\\Helipad_DataBase_meta\\Helipad_DataBase_meta_original"
    # model_folder = "C:\\Users\\jonas\\Desktop\\Helipad\\model\\helipad_cfg20191126T2346"
    # weight_filename = "mask_rcnn_helipad_cfg_0088.h5"

    image_folder = "../../Helipad_DataBase/Helipad_DataBase_original"
    meta_folder = "../../Helipad_DataBase_meta/Helipad_DataBase_meta_original"
    model_root_folder = "../../model/"
    model_folder = "helipad_cfg_9_no47_aug2_3+20200112T2326"
    model_number = 8
    weight_filename = "mask_rcnn_helipad_cfg_9_no47_aug2_3+_0257.h5"

    # test_only = True
    activate_filter = False

    run_detection = RunDetection(image_folder,
                              meta_folder,
                              os.path.join(model_root_folder, model_folder),
                              weight_filename,
                              model_number=model_number,
                              activate_filter=activate_filter)

    run_detection.run()

    # RunDetection.review_prediction(image_folder,
    #                                meta_folder,
    #                                model_number,
    #                                test_only)

    # image_folder = "../Helipad_DataBase/Helipad_DataBase_original"
    # meta_folder = "../Helipad_DataBase_meta/Helipad_DataBase_meta_original"
    #
    # model_root_folder = "../model/"
    # model_folders = ["helipad_cfg20191126T2346",
    #                  "helipad_cfg_aug220191209T1456",
    #                  "helipad_cfg_aug320191210T2238",
    #                  "helipad_cfg_aug2_5+20191211T1749"]
    # weight_filenames = ["mask_rcnn_helipad_cfg_0088.h5",
    #                     "mask_rcnn_helipad_cfg_aug2_0209.h5",
    #                     "mask_rcnn_helipad_cfg_aug3_0228.h5",
    #                     "mask_rcnn_helipad_cfg_aug2_5+_0381.h5"]
    #
    # model_numbers = [1, 2, 3, 4]
    #
    # activate_filter = False
    #
    # for i in range(len(model_numbers)):
    #     print("Model {}".format(i+1))
    #     run_detection = RunDetection(image_folder,
    #                                  meta_folder,
    #                                  os.path.join(model_root_folder, model_folders[i]),
    #                                  weight_filenames[i],
    #                                  model_number=model_numbers[i],
    #                                  activate_filter=activate_filter)
    #
    #     run_detection.run()





