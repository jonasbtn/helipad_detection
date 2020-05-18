import os
import json
import numpy as np
import pandas as pd
from time import time

import sys
sys.path.append('../')

from helipad_detection.src.detection.run_detection import RunDetection
from helipad_detection.src.training.filter_manager import FilterManager

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class BenchmarkManager:

    def __init__(self, image_folder, meta_folder,
                 test_only=True, tms_dataset=False, zoom_level=None,
                 include_category=None,
                 include_negative=True,
                 city_lat=None,
                 train_only=False):

        self.image_folder = image_folder
        self.meta_folder = meta_folder

        print("Loading Files")
        self.target_files = RunDetection.build_target_files(self.image_folder,
                                                            self.meta_folder,
                                                            test_only=test_only,
                                                            tms_dataset=tms_dataset,
                                                            zoom_level=zoom_level,
                                                        include_category=include_category,
                                                            include_negative=include_negative,
                                                            city_lat=city_lat,
                                                            train_only=train_only)
        print("{} files loaded!".format(len(self.target_files)))

    def reinitialize_metrics(self):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

        self.metrics_per_categories = {}
        for i in range(10):
            self.metrics_per_categories[str(i)] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        self.metrics_per_categories['u'] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        self.metrics_per_categories['d'] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

    @staticmethod
    def check_false_positive(groundtruth, predicted, threshold_iou=0.5, threshold_area=0.8):
        # if no box detected
        if "box" not in predicted:
            return 0
        # if some boxes detected but none in the groundtruth
        # there are len(boxes) false positive
        if "box" not in groundtruth or len(groundtruth["box"]) == 0:
            return len(predicted["box"])

        bboxes_predicted = predicted["box"]
        bboxes_groundtruth = groundtruth["box"]

        nb_FN = 0

        # for all predicted
        for j in range(len(bboxes_predicted)):
            box_predicted = bboxes_predicted[j]
            IOUs = []
            contains = [False]*len(bboxes_groundtruth)

            # compute iou between predicted and each bbox groundtruth
            for k in range(len(bboxes_groundtruth)):
                box_groundtruth = bboxes_groundtruth[k]

                interArea = FilterManager.compute_interArea(box_predicted, box_groundtruth)
                boxAArea = FilterManager.compute_area(box_predicted)
                boxBArea = FilterManager.compute_area(box_groundtruth)
                iou = interArea / float(boxAArea + boxBArea - interArea)

                if interArea > boxAArea*threshold_area or interArea > boxBArea*threshold_area:
                    contains[k] = True

                IOUs.append(iou)

            # if max iou < threshold, and the predicted box is not contained inside a groundtruth
            # classify the predicted box as false positive
            arg_max_IOUs = np.argmax(IOUs)

            if IOUs[arg_max_IOUs] < threshold_iou and not contains[arg_max_IOUs]:
                nb_FN += 1

        return nb_FN

    @staticmethod
    def check_true_positive(groundtruth, predicted, threshold_iou=0.5, threshold_area=0.8):
        # if no box detected
        if "box" not in predicted or len(predicted["box"]) == 0:
            return 0
        # need to fix this, why are there helipad without box ?
        if "box" not in groundtruth or len(groundtruth["box"]) == 0:
            return 0

        bboxes_predicted = predicted["box"]
        bboxes_groundtruth = groundtruth["box"]

        nb_TP = 0

        # for all predicted
        for j in range(len(bboxes_predicted)):
            box_predicted = bboxes_predicted[j]
            IOUs = []
            contains = [False] * len(bboxes_groundtruth)

            # compute iou for each bbox groundtruth
            for k in range(len(bboxes_groundtruth)):
                box_groundtruth = bboxes_groundtruth[k]

                interArea = FilterManager.compute_interArea(box_predicted, box_groundtruth)
                boxAArea = FilterManager.compute_area(box_predicted)
                boxBArea = FilterManager.compute_area(box_groundtruth)
                iou = interArea / float(boxAArea + boxBArea - interArea)

                if interArea > boxAArea*threshold_area or interArea > boxBArea*threshold_area:
                    contains[k] = True

                IOUs.append(iou)

            arg_max_IOUs = np.argmax(IOUs)

            # if max iou > threshold or if the predicted box is contained inside a groundtruth
            # classify the predicted box as true positive
            if IOUs[arg_max_IOUs] > threshold_iou or contains[arg_max_IOUs]:
                nb_TP += 1

        return nb_TP

    @staticmethod
    def filter_predicted(predicted, threshold_score, threshold_iou, threshold_area, threshold_validation=None):
        if "box" not in predicted or len(predicted["box"]) == 0:
            return predicted
        bboxes = predicted["box"]
        class_ids = predicted["class_id"]
        scores = predicted["score"]

        if threshold_validation:
            scores_validation = predicted["cnn_validation"]
        else:
            scores_validation = None

        # Filter overlapping box (see FilterManager for default value of threshold_iou and threshold_area)
        # bboxes, class_ids, scores = FilterManager.filter_by_iou(bboxes,
        #                                                         class_ids,
        #                                                         scores,
        #                                                         threshold_iou=threshold_iou,
        #                                                         threshold_area=threshold_area)

        # remove if score < threshold (see FilterManager for default values)
        bboxes, class_ids, scores = FilterManager.filter_by_scores(bboxes,
                                                                   class_ids,
                                                                   scores,
                                                                   threshold=threshold_score,
                                                                   threshold_validation=threshold_validation,
                                                                   scores_validation=scores_validation)

        predicted["box"] = bboxes
        predicted["class_id"] = class_ids
        predicted["score"] = scores
        if len(bboxes) == 0:
            predicted["helipad"] = False

        return predicted

    def run(self, model_number, threshold_score, threshold_iou, threshold_area, threshold_validation=None):

        self.model_number = model_number

        self.reinitialize_metrics()

        L = len(self.target_files)

        for i in range(L):

            image_meta_path = self.target_files[i]
            imagepath = image_meta_path[0]
            metapath = image_meta_path[1]

            with open(metapath, 'r') as f:
                meta = json.load(f)
            f.close()

            if "groundtruth" not in meta:
                continue
            if "predicted" not in meta:
                continue

            key = "model_{}".format(model_number)

            if key not in meta["predicted"]:
                print("Model not predicted yet")
                break

            groundtruth = meta["groundtruth"].copy()
            predicted = meta["predicted"][key].copy()

            # Apply filtering here
            predicted_filtered = self.filter_predicted(predicted, threshold_score, threshold_iou, threshold_area,
                                                       threshold_validation=threshold_validation)

            # if not groundtruth["helipad"] and predicted["helipad"]:
            FP = self.check_false_positive(groundtruth, predicted_filtered, threshold_iou=0.5, threshold_area=0.8)
            self.FP += FP
            if "category" in groundtruth:
                self.metrics_per_categories[groundtruth["category"]]['FP'] += FP

            # if groundtruth["helipad"] and predicted["helipad"]:
            TP = self.check_true_positive(groundtruth, predicted_filtered, threshold_iou=0.5, threshold_area=0.8)
            self.TP += TP
            if "category" in groundtruth:
                self.metrics_per_categories[groundtruth["category"]]['TP'] += TP

            if not groundtruth["helipad"] and not predicted_filtered["helipad"]:
                self.TN += 1
                if "category" in groundtruth:
                    self.metrics_per_categories[groundtruth["category"]]['TN'] += 1

            if groundtruth["helipad"] and "box" in groundtruth and not predicted_filtered["helipad"]:
                self.FN += len(groundtruth["box"])
                if "category" in groundtruth:
                    self.metrics_per_categories[groundtruth["category"]]['FN'] += len(groundtruth["box"])

        # print(self.TP)
        # print(self.TN)
        # print(self.FP)
        # print(self.FN)

        self.accuracy = (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
        self.error = (self.FP+self.FN) / (self.TP + self.TN + self.FP + self.FN)
        if (self.TP + self.FP)>0:
            self.precision = self.TP / (self.TP + self.FP)
        else:
            self.precision = 0
        if (self.TP + self.FN)>0:
            self.recall = self.TP / (self.TP + self.FN)
        else:
            self.recall = 0
        if (self.TN + self.FP)>0:
            self.FPR = self.FP / (self.TN + self.FP)
        else:
            self.FPR = 0
        self.TPR = self.recall

        data = [model_number, threshold_score, threshold_iou, threshold_area,
                self.accuracy, self.error, self.precision, self.recall,
                self.FPR, self.TPR, self.TP, self.TN, self.FP, self.FN]

        return data

    def save_benchmark(self):
        with open("benchmark_model_{}.txt".format(self.model_number), 'w') as f:
            f.write("Accuracy : {}\nError : {}\nPrecision : {}\nRecall : {}\n".format(
                self.accuracy,
                self.error,
                self.precision,
                self.recall
            ))
            f.write("True Positive : {}\nTrue Negative : {}\nFalse Positive : {}\nFalse Negative : {}\n".format(
                self.TP,
                self.TN,
                self.FP,
                self.FN
            ))
            f.write(json.dumps(self.metrics_per_categories, indent=4, sort_keys=True))
            f.close()

    def get_attributes(self):
        data = [self.model_number, self.accuracy, self.error, self.precision, self.recall,
                self.FPR, self.TPR, self.TP, self.TN, self.FP, self.FN]
        return data


if __name__ == "__main__":

    image_folder = "C:\\Users\\jonas\\Desktop\\Helipad\\Helipad_DataBase\\Helipad_DataBase_original"
    meta_folder = "C:\\Users\\jonas\\Desktop\\Helipad\\Helipad_DataBase_meta\\Helipad_DataBase_meta_original"

    # image_folder = "../Helipad_DataBase/Helipad_DataBase_original"
    # meta_folder = "../Helipad_DataBase_meta/Helipad_DataBase_meta_original"

    tms_image_folder = "C:\\Users\\jonas\\Desktop\\Real_World_Test_DataBase"
    tms_meta_folder = "C:\\Users\\jonas\\Desktop\\Real_World_Test_DataBase_meta"
    model_number = 5

    threshold_score = 0
    threshold_iou = 0.5
    threshold_area = 0.8

    # if model_number:
    benchmark_manager = BenchmarkManager(tms_image_folder,
                                         tms_meta_folder,
                                         test_only=True,
                                         tms_dataset=True)

    result = benchmark_manager.run(model_number, threshold_score, threshold_iou, threshold_area)
    print(result)












