##
import os
import json

from numpy import zeros
from numpy import asarray

from mrcnn.utils import Dataset
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes

from mrcnn.model import MaskRCNN


# add image size to meta

class HelipadDataset(Dataset):

    def load_dataset(self, root_folder, root_meta_folder, is_train=True, include_augmented=False, augmented_versions=[],
                     include_categories=None):
        self.add_class("dataset", 1, "helipad")

        image_original_folder = os.path.join(root_folder, 'Helipad_DataBase_original')
        meta_original_folder = os.path.join(root_meta_folder, 'Helipad_DataBase_meta_original')

        for subdir, dirs, files in os.walk(image_original_folder):
            for file in files:
                if file[0] == ".":
                    continue
                image_path = os.path.join(subdir, file)
                meta_filepath = os.path.join(meta_original_folder,
                                             os.path.basename(subdir),
                                             os.path.splitext(file)[0]+".meta")
                image_name = os.path.splitext(file)[0]
                image_number = int(image_name.split('_')[1])

                with open(meta_filepath, 'r') as f:
                    meta = json.load(f)
                if not "groundtruth" in meta:
                    continue
                elif not meta["groundtruth"]["helipad"]:
                    continue
                # or add the false positive here as not helipad ?
                elif "box" not in meta["groundtruth"]:
                    continue


                # elif include_categories:
                #     if meta["groundtruth"]["category"] not in include_categories:
                #         continue
                # else:
                #     # change to add shuffle to change the train set
                #     # or not ?
                #     if is_train and image_number > 4250:
                #         continue
                #     if not is_train and image_number <= 4250:
                #         continue

                if is_train and image_number > 4250:
                    continue
                if not is_train and image_number <= 4250:
                    continue

                if include_categories:
                    if meta["groundtruth"]["category"] not in include_categories:
                        continue

                self.add_image('dataset',
                               image_id=os.path.splitext(file)[0],
                               path=image_path,
                               annotation=meta_filepath)

        if is_train and include_augmented:
            for version in augmented_versions:
                image_aug_folder = os.path.join(root_folder, 'Helipad_DataBase_augmented_{}'.format(version))
                meta_aug_folder = os.path.join(root_meta_folder, 'Helipad_DataBase_meta_augmented_{}'.format(version))
                for subdir, dirs, files in os.walk(image_aug_folder):
                    for file in files:
                        image_path = os.path.join(subdir, file)
                        meta_filepath = os.path.join(meta_aug_folder,
                                                     os.path.basename(subdir),
                                                     os.path.splitext(file)[0]+".meta")
                        with open(meta_filepath, 'r') as f:
                            meta = json.load(f)
                        if not "groundtruth" in meta:
                            continue
                        elif not meta["groundtruth"]["helipad"]:
                            continue
                        elif "box" not in meta["groundtruth"]:
                            continue
                        # elif include_categories:
                        #     if meta["groundtruth"]["category"] not in include_categories:
                        #         continue
                        # else:
                        #     image_name = os.path.splitext(file)[0]
                        #     image_number = int(image_name.split('_')[1])
                        #     if image_number > 4250:
                        #         continue

                        image_name = os.path.splitext(file)[0]
                        image_number = int(image_name.split('_')[1])
                        if image_number > 4250:
                            continue
                        if include_categories:
                            if meta["groundtruth"]["category"] not in include_categories:
                                continue

                        self.add_image('dataset',
                                       image_id=os.path.splitext(file)[0]+"_v{}".format(version),
                                       path=image_path,
                                       annotation=meta_filepath)

    def extract_bboxes(self, meta_filepath):
        with open(meta_filepath, 'r') as f:
            meta = json.load(f)
        if "groundtruth" not in meta:
            return []
        elif "box" not in meta["groundtruth"]:
            return []
        else:
            meta_bboxes = meta["groundtruth"]["box"]
            bboxes = []
            for box in meta_bboxes:
                min_x = min(box[0], box[2])
                min_y = min(box[1], box[3])
                max_x = max(box[0], box[2])
                max_y = max(box[1], box[3])
                # min_x = box[0]
                # min_y = box[1]
                # max_x = box[2]
                # max_y = box[3]
                bboxes.append([min_y, min_x, max_y, max_x])

            return bboxes, 640, 640

    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['annotation']
        # load meta
        boxes, w, h = self.extract_bboxes(path)
        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[0], box[2]
            col_s, col_e = box[1], box[3]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('helipad'))
        return masks, asarray(class_ids, dtype='int32')

    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']