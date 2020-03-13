import os
import json
import cv2
from tqdm import tqdm as tqdm
import numpy as np

from src.training.filter_manager import FilterManager


class BBBuildDataset:

    def __init__(self, image_folder, meta_folder, model_number,
                 score_threshold, iou_threshold,
                 output_folder, tms=False,
                 groundtruth_bb = True,
                 filter_categories=None):
        self.image_folder = image_folder
        self.meta_folder = meta_folder
        self.model_number = model_number
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.output_folder = output_folder
        self.tms = tms
        # ground_truth_bb indicates wheter the dataset keeps the groundtruth
        # as the true positive (True) or the true positive detected by the model (False)
        # Sometime, the detection by the model is not centered in the helipad
        # hence it can add noise to the model
        self.groundtruth_bb = groundtruth_bb
        self.filter_categories = filter_categories

    def build_target_files(self):
        target_files = []
        for subdir, dirs, files in os.walk(self.meta_folder, topdown=True):
            for file in files:
                meta_path = os.path.join(subdir, file)
                if not self.tms:
                    image_path = os.path.join(self.image_folder,
                                              os.path.basename(subdir),
                                              os.path.splitext(file)[0]+".png")
                else:
                    image_info = os.path.splitext(file)[0].split("_")
                    zoom = image_info[1]
                    xtile = image_info[2]
                    ytile = image_info[3]
                    image_path = os.path.join(self.image_folder,
                                              xtile,
                                              str(ytile)+".jpg")
                target_files.append([image_path, meta_path])
        return target_files

    def get_output_file_name(self, classe, image_path, box_id):
        """
        output_folder/model_{number}_{score}/classes/folder_id/image_name_{}
        TODO: Support TMS Dataset (use meta_path)
        :param classe:
        :param image_path:
        :param box_id:
        :return:
        """
        image_name, ext = os.path.splitext(os.path.basename(image_path))
        folder_name = os.path.basename(os.path.dirname(image_path))
        folder_id = int(folder_name.split('_')[1])
        dataset = "train"
        if classe == "false_positive":
            if folder_id > 44:
                dataset = "test"
        elif classe == "helipad":
            if folder_id > 48:
                dataset = "test"
        if not os.path.isdir(self.output_folder):
            os.mkdir(self.output_folder)
        elif not os.path.isdir(os.path.join(self.output_folder,
                                            f'model_{self.model_number}_{self.score_threshold}')):
            os.mkdir(os.path.join(self.output_folder,
                                            f'model_{self.model_number}_{self.score_threshold}'))
        elif not os.path.isdir(os.path.join(self.output_folder,
                                            f'model_{self.model_number}_{self.score_threshold}',
                                            dataset)):
            os.mkdir(os.path.join(self.output_folder,
                                            f'model_{self.model_number}_{self.score_threshold}',
                                            dataset))

        elif not os.path.isdir(os.path.join(self.output_folder,
                                            f'model_{self.model_number}_{self.score_threshold}',
                                            dataset,
                                            classe)):
            os.mkdir(os.path.join(self.output_folder,
                                  f'model_{self.model_number}_{self.score_threshold}',
                                  dataset,
                                  classe))
        # elif not os.path.isdir(os.path.join(self.output_folder,
        #                                     f'model_{self.model_number}_{self.score_threshold}',
        #                                     dataset,
        #                                     classe,
        #                                     folder_name)):
        #     os.mkdir(os.path.join(self.output_folder,
        #                           f'model_{self.model_number}_{self.score_threshold}',
        #                           dataset,
        #                           classe,
        #                           folder_name))
        # output_path = os.path.join(self.output_folder,
        #                            f'model_{self.model_number}_{self.score_threshold}',
        #                            classe,
        #                            folder_name,
        #                            image_name+"_"+str(box_id)+ext)
        output_path = os.path.join(self.output_folder,
                                   f'model_{self.model_number}_{self.score_threshold}',
                                   dataset,
                                   classe,
                                   image_name+"_"+str(box_id)+ext)
        return output_path

    def get_output_file_name_tms(self, image_path, box_id):
        """
        output_folder/model_{number}_{score}/tms/xtile/Satellite_{zoom}_{xtile}_{ytile}_{box_id}.jpg
        :param classe:
        :param image_path:
        :param box_id:
        :return:
        """
        ytile = os.path.splitext(os.path.basename(image_path))[0]
        xtile = os.path.basename(os.path.dirname(image_path))
        zoom = os.path.basename(os.path.dirname(os.path.dirname(image_path)))

        if not os.path.isdir(self.output_folder):
            os.mkdir(self.output_folder)
        elif not os.path.isdir(os.path.join(self.output_folder,
                                            f'model_{self.model_number}_{self.score_threshold}')):
            os.mkdir(os.path.join(self.output_folder,
                                            f'model_{self.model_number}_{self.score_threshold}'))
        elif not os.path.isdir(os.path.join(self.output_folder,
                                            f'model_{self.model_number}_{self.score_threshold}',
                                            "tms")):
            os.mkdir(os.path.join(self.output_folder,
                                            f'model_{self.model_number}_{self.score_threshold}',
                                            "tms"))
        elif not os.path.isdir(os.path.join(self.output_folder,
                                            f'model_{self.model_number}_{self.score_threshold}',
                                            "tms",
                                            xtile)):
            os.mkdir(os.path.join(self.output_folder,
                                            f'model_{self.model_number}_{self.score_threshold}',
                                            "tms",
                                            xtile))
        output_path = os.path.join(self.output_folder,
                                   f'model_{self.model_number}_{self.score_threshold}',
                                   "tms",
                                   xtile,
                                   f'Satellite_{zoom}_{xtile}_{ytile}_{box_id}.jpg')
        return output_path

    def save_image(self, image_box, output_path):
        # print(output_path)
        # print(image_box)
        cv2.imwrite(output_path, image_box)

    def detect_false_postive(self, box_predicted, bboxes_groundtruth):
        false_positive = False
        contains = False
        # detect if box is a true positive or a false positive
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

                # check if box in contained:
                boxA = box_predicted
                boxB = box_groundtruth
                # if boxA is contained inside boxB
                if boxA[0] > boxB[0] and boxA[1] > boxB[1] and boxA[2] < boxB[2] and boxA[3] < boxB[3]:
                    contains = True
                # if boxB is contained inside boxA
                elif boxA[0] < boxB[0] and boxA[1] < boxB[1] and boxA[2] > boxB[2] and boxA[3] > boxB[3]:
                    contains = True

            # if max iou < threshold, and the predicted box is not contained inside a groundtruth
            # classify the predicted box as false positive
            arg_max_IOUs = np.argmax(IOUs)

            if IOUs[arg_max_IOUs] < self.iou_threshold:
                false_positive = True
        else:
            false_positive = True

        return false_positive, contains

    def run(self):

        self.target_files = self.build_target_files()

        for i in tqdm(range(len(self.target_files))):
            image_meta_path = self.target_files[i]
            image_path = image_meta_path[0]
            meta_path = image_meta_path[1]

            with open(meta_path, 'r') as f:
                meta = json.load(f)
            f.close()

            image = cv2.imread(image_path)

            if image is None:
                print(image_path)
                continue

            if not self.tms and self.groundtruth_bb and "groundtruth" not in meta:
                continue
            if "predicted" not in meta:
                continue
            elif "model_{}".format(self.model_number) not in meta["predicted"]:
                continue

            box_id = 0

            if not self.tms:
                groundtruth = meta["groundtruth"]

                if groundtruth["helipad"]:
                    if "box" in groundtruth:
                        bboxes_groundtruth = groundtruth["box"]
                    else:
                        bboxes_groundtruth = []

                else:
                    bboxes_groundtruth = []

                if self.groundtruth_bb:
                    # keep the groundtruth for the true positives
                    for box_groundtruth in bboxes_groundtruth:
                        x_min = min(box_groundtruth[0], box_groundtruth[2])
                        y_min = min(box_groundtruth[1], box_groundtruth[3])
                        x_max = max(box_groundtruth[2], box_groundtruth[0])
                        y_max = max(box_groundtruth[3], box_groundtruth[1])
                        image_box = image[y_min:y_max, x_min:x_max, :]

                        if self.filter_categories and "category" in groundtruth:
                            if groundtruth["category"] in self.filter_categories:
                                # put it as false positive
                                output_classe = "false_positive"
                            else:
                                output_classe = "helipad"
                        else:
                            output_classe = "helipad"
                        # get the name of the output_file
                        output_path = self.get_output_file_name(output_classe,
                                                                image_path,
                                                                box_id)
                        # save the file
                        self.save_image(image_box, output_path)
                        box_id += 1

            predicted = meta["predicted"][f'model_{self.model_number}']
            bboxes_predicted = predicted["box"]
            scores_predicted = predicted["score"]

            for j in range(len(bboxes_predicted)):

                box_predicted = bboxes_predicted[j]
                score_predicted = scores_predicted[j]

                if score_predicted < self.score_threshold:
                    continue

                x_min = box_predicted[0]
                y_min = box_predicted[1]
                x_max = box_predicted[2]
                y_max = box_predicted[3]

                image_box = image[y_min:y_max, x_min:x_max, :]

                if self.tms:
                    #TODO: Add support for tms
                    output_path = self.get_output_file_name_tms(image_path, box_id)
                    self.save_image(image_box, output_path)
                    continue

                false_positive, contains = self.detect_false_postive(box_predicted, bboxes_groundtruth)

                if self.groundtruth_bb:
                    if contains:
                        continue
                    elif false_positive:
                        # keep only the false positive
                        output_path = self.get_output_file_name("false_positive",
                                                                image_path,
                                                                box_id)
                        self.save_image(image_box, output_path)
                        box_id += 1
                else:
                    # keep all detected boxes and label them as true or false
                    if contains:
                        classe = "helipad"
                    elif false_positive:
                        classe = "false_positive"
                    else:
                        classe = "helipad"

                    output_path = self.get_output_file_name("false_positive",
                                                            image_path,
                                                            box_id)
                    self.save_image(image_box, output_path)
                    box_id += 1


if __name__ == "__main__":
    image_folder = "C:\\Users\\jonas\\Desktop\\Helipad\\Helipad_DataBase\\Helipad_DataBase_original"
    meta_folder = "C:\\Users\\jonas\\Desktop\\Helipad\\Helipad_DataBase_meta\\Helipad_DataBase_meta_original"
    model_number = 7
    score_threshold = 0.0
    iou_threshold = 0.5
    output_folder = "C:\\Users\\jonas\\Desktop\\Helipad\\Detected_Boxes_3"
    tms = False
    groundtruth_bb = True
    filter_categories = ["4", "7", "d", "u"]

    bb_build_dataset = BBBuildDataset(image_folder=image_folder,
                                      meta_folder=meta_folder,
                                      model_number=model_number,
                                      score_threshold=score_threshold,
                                      iou_threshold=iou_threshold,
                                      output_folder=output_folder,
                                      tms=tms,
                                      groundtruth_bb=groundtruth_bb,
                                      filter_categories=filter_categories)

    bb_build_dataset.run()
    #
    # #TODO: Include Categories (skip 4 and 7 for example)

    # image_folder = "C:\\Users\\AISG\\Documents\\Jonas\\Real_World_Dataset_TMS\\sat\\19"
    # meta_folder = "C:\\Users\\AISG\\Documents\\Jonas\\Real_World_Dataset_TMS_meta\\sat\\19"
    # model_number = 7
    # score_threshold = 0.0
    # iou_threshold = 0.5
    # output_folder = "C:\\Users\\AISG\\Documents\\Jonas\\Detected_Boxes"
    # tms = True
    # groundtruth_bb = False
    #
    # bb_build_dataset = BBBuildDataset(image_folder=image_folder,
    #                                   meta_folder=meta_folder,
    #                                   model_number=model_number,
    #                                   score_threshold=score_threshold,
    #                                   iou_threshold=iou_threshold,
    #                                   output_folder=output_folder,
    #                                   tms=tms,
    #                                   groundtruth_bb=groundtruth_bb)
    #
    # bb_build_dataset.run()
