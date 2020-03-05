import os
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.training.filter_manager import FilterManager


class KNNBuildDatabase:

    def __init__(self, image_folder, meta_folder, model_number, train=True, TMS=False):

        self.image_folder = image_folder
        self.meta_folder = meta_folder
        self.model_number = model_number
        self.train = train
        self.TMS = TMS
        self.X = []
        self.y = []
        self.image_id = []

    @staticmethod
    def convert_cat_str_to_int(str_cat):
        for i in range(10):
            if str_cat == str(i):
                return i
        if str_cat == "d":
            return 10
        elif str_cat == "u":
            return 11

    def run(self):

        for subdir, dirs, files in os.walk(self.meta_folder, topdown=True):
            for file in files:
                with open(os.path.join(subdir, file), 'r') as f:
                    meta = json.load(f)

                if not self.TMS:
                    image = cv2.imread(os.path.join(self.image_folder,
                                                    os.path.basename(subdir),
                                                    os.path.splitext(file)[0]+".png"))
                else:
                    image_info = os.path.splitext(file)[0].split("_")
                    zoom = image_info[1]
                    xtile = image_info[2]
                    ytile = image_info[3]
                    image = cv2.imread(os.path.join(self.image_folder,
                                                    zoom,
                                                    xtile,
                                                    str(ytile)+".jpg"))

                if self.train:
                    if "groundtruth" not in meta:
                        continue
                elif "predicted" not in meta:
                    continue
                elif "model_{}".format(self.model_number) not in meta["predicted"]:
                    continue

                predicted = meta["predicted"][f'model_{self.model_number}']
                bboxes_predicted = predicted["box"]

                if self.train:
                    groundtruth = meta["groundtruth"]

                    if groundtruth["helipad"]:
                        if "box" in groundtruth:
                            bboxes_groundtruth = groundtruth["box"]
                        else:
                            bboxes_groundtruth = []
                    else:
                        bboxes_groundtruth = []

                for box_predicted in bboxes_predicted:
                    x_min = box_predicted[0]
                    y_min = box_predicted[1]
                    x_max = box_predicted[2]
                    y_max = box_predicted[3]

                    image_box = image[x_min:x_max, y_min:y_max, :]

                    if self.train:
                        target = -1
                        if predicted["helipad"] and not groundtruth["helipad"]:
                            # false positive
                            target = 12
                        elif groundtruth["helipad"] and predicted["helipad"]:

                            if len(bboxes_groundtruth) > 0:
                                # check IOU
                                IOUs = []

                                # check IOU with groundtruth
                                # compute iou between predicted and each bbox groundtruth
                                for k in range(len(bboxes_groundtruth)):
                                    box_groundtruth = bboxes_groundtruth[k]

                                    interArea = FilterManager.compute_interArea(box_predicted, box_groundtruth)
                                    boxAArea = FilterManager.compute_area(box_predicted)
                                    boxBArea = FilterManager.compute_area(box_groundtruth)
                                    iou = interArea / float(boxAArea + boxBArea - interArea)

                                    IOUs.append(iou)

                                # if max iou < threshold, and the predicted box is not contained inside a groundtruth
                                # classify the predicted box as false positive
                                arg_max_IOUs = np.argmax(IOUs)

                                if IOUs[arg_max_IOUs] < 0.5:
                                    # false positive
                                    target = 12
                                else:
                                    if "category" in groundtruth:
                                        target = self.convert_cat_str_to_int(groundtruth["category"])
                            else:
                                if "category" in groundtruth:
                                    target = self.convert_cat_str_to_int(groundtruth["category"])

                        if target == -1:
                            continue
                        else:
                            self.y.append(target)

                    self.X.append(image_box)
                    self.image_id.append(os.path.splitext(file)[0])


if __name__ == "__main__":

    image_folder = "C:\\Users\\jonas\\Desktop\\Helipad\\Helipad_DataBase\\Helipad_DataBase_original"
    meta_folder = "C:\\Users\\jonas\\Desktop\\Helipad\\Helipad_DataBase_meta\\Helipad_DataBase_meta_original"
    model_number = 7

    knn_build_database = KNNBuildDatabase(image_folder, meta_folder, model_number)

    knn_build_database.run()

