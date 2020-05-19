import os
import json
import cv2
import time
import argparse
from tqdm import tqdm

import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf

tf.enable_eager_execution()

# from imgaug import augmenters as iaa

from helipad_detection.src.utils.autoaugment_utils import *
from helipad_detection.src.utils.box_utils import *


class DatabaseAugmentation:
    """
    Apply Augmentation on the dataset using Google's policy. \n
    The policy is editable inside the script file `helipad_detection.src.utils.autoaugment_utils` under the function `policy_v3`.
    """
    def __init__(self, input_folder, meta_folder, root_folder, root_folder_meta,
                 balance_dataset=False, repartition=None, version_number=None, display=False):
        """
        `input_folder`: the folder containing the original images \n
        `meta_folder`: the folder containing the meta of the original images \n
        `root_folder`: the folder where to store the augmented images \n
        `root_folder_meta`: the folder where to store the meta of the augmented images \n
        `balance_dataset`: boolean, if yes, the number of images per category is balanced by augmenting more the images from small categories \n
        `repartition`: list of 12 integers precising the number of times each image per category is augmented  \n
        `version_number`: the augmentation version number. The suffix of the output folder will be the version number \n 
        `display`: boolean to display the augmented images as the script augment them \n
        """
        self.input_folder = input_folder
        self.meta_folder = meta_folder
        self.root_folder = root_folder
        self.root_folder_meta = root_folder_meta
        self.version_number = version_number
        self.aug_foldername = self.set_aug_foldername(self.root_folder, version_number=self.version_number)
        self.output_folder = os.path.join(root_folder, self.aug_foldername)
        self.aug_meta_foldername = self.set_aug_foldername(self.root_folder_meta, version_number=self.version_number)
        self.meta_output_folder = os.path.join(os.path.dirname(self.meta_folder), self.aug_meta_foldername)

        if not os.path.isdir(self.output_folder):
            os.mkdir(self.output_folder)
        if not os.path.isdir(self.meta_output_folder):
            os.mkdir(self.meta_output_folder)

        self.balance_dataset = balance_dataset
        self.repartition = repartition

        print("Building Target files")
        if self.balance_dataset:
            self.target_files = self.balance_categories(self.input_folder, self.meta_folder, repartition)
        elif repartition:
            self.target_files = self.duplicate_categories(self.input_folder, self.meta_folder, repartition)
        else:
            self.target_files = self.build_target_files()
        print("Target Files Built")
        print("Generating {} images".format(len(self.target_files)))

        if display == "True":
            self.display = True
        elif display == "False":
            self.display = False
        else:
            self.display = display

        print(type(self.display))
        print("Displaying samples : {}".format(self.display))
        self.sess = tf.InteractiveSession()

    @staticmethod
    def set_aug_foldername(folder, version_number=None):
        """
        Set the augmented folder name with the suffix `version_number`. The output folder is a sub-directory of `folder`. 
        """
        if not version_number:
            directories = os.listdir(folder)
            print(directories)
            i = 1
            for dir in directories:
                if dir[0] == '.':
                    continue
                elements = dir.split('_')
                if elements[-2] == 'augmented':
                    i += 1
            aug_foldername = os.path.basename(folder)+"_augmented_{}".format(i)
        else:
            aug_foldername = os.path.basename(folder)+"_augmented_{}".format(version_number)
        return aug_foldername

    @staticmethod
    def set_aug_filename(filename, i):
        """
        Set the augmentation filename using the `filename` and its augmentation id `i`.
        """
        filename_ext = os.path.splitext(filename)
        aug_filename = filename_ext[0]+"_aug_{:03d}".format(i)+filename_ext[1]
        return aug_filename

    @staticmethod
    def categories_imagemeta_path(input_folder, meta_folder):
        """
        Returns a dictionnary having a category as a key and a list of tuples (image_path, meta_filepath) belonging to this category as value.
        """
        categories_path = {}
        for subdir, dirs, files in os.walk(input_folder, topdown=True):
            for file in files:
                image_path = os.path.join(subdir, file)
                meta_filepath = os.path.join(meta_folder,
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
                elif "category" not in meta["groundtruth"]:
                    continue

                category = meta["groundtruth"]["category"]

                if category not in categories_path:
                    categories_path[category] = [[image_path, meta_filepath]]
                else:
                    categories_path[category].append([image_path, meta_filepath])
        return categories_path

    @staticmethod
    def duplicate_categories(input_folder, meta_folder, repartition):
        """
        Duplicate the categories respectively to the `repartition` chosen\n
        Return a list of target files
        """
        categories_path = DatabaseAugmentation.categories_imagemeta_path(input_folder, meta_folder)
        target_files = []

        for key, value in categories_path.items():
            if "0" <= key <= "9":
                cat = int(key)
            if key == "d":
                cat = 10
            if key == "u":
                cat = 11

            for k in range(repartition[cat]):
                target_files.extend(value)

        return target_files

    @staticmethod
    def balance_categories(input_folder, meta_folder, repartition=None):
        """
        Balance the categories first and then apply the `repartition` chosen \n
        Return a list of target files. 
        """
        categories_path = DatabaseAugmentation.categories_imagemeta_path(input_folder, meta_folder)
        categories_count = []
        categories_count_dict = {}
        for key, value in categories_path.items():
            categories_count.append(len(value))
            categories_count_dict[key] = len(value)
        print("Categories count : ", end="")
        print(categories_count)
        print(categories_count_dict)
        max_count = np.amax(categories_count)
        print("Max count : ", end="")
        print(max_count)
        total_count = np.sum(categories_count)
        print("Total count : ", end="")
        print(total_count)

        target_files = []

        for key, value in categories_path.items():
            target_files.extend(value)
            L = len(value)

            nb_image_to_balance = max_count-L

            # if nb_image_to_balance <= L:
            #     picked_index = np.random.choice(L, nb_image_to_balance, replace=False)
            #     picked = list(np.array(value)[picked_index])
            # else:
            picked_index = np.random.choice(L, nb_image_to_balance, replace=True)
            picked = list(np.array(value)[picked_index])
            target_files.extend(picked)

            # After Balance, duplicate categories
            if repartition:
                if "0" <= key <= "9":
                    cat_int = int(key)
                elif key == "d":
                    cat_int = 10
                elif key == "u":
                    cat_int = 11

                nb_duplicate = repartition[cat_int]
                print("Duplicating category {} {} times".format(key, nb_duplicate))

                for i in range(nb_duplicate):
                    picked_index = np.random.choice(L, max_count, replace=True)
                    picked = list(np.array(value)[picked_index])
                    target_files.extend(picked)

        return target_files

    def build_target_files(self):
        """
        Return a list of target files from the `input_folder`.
        """
        target_files = []

        for subdir, dirs, files in os.walk(self.input_folder, topdown=True):
            for file in files:
                meta_filepath = os.path.join(self.meta_folder,
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

                bboxes = meta["groundtruth"]["box"]

                if len(bboxes) == 0:
                    continue

                target_files.append([os.path.join(subdir, file), meta_filepath])

        return target_files

    def run(self):
        """
        Run the augmentation on the dataset after initialization
        """

        for i in tqdm(range(len(self.target_files))):

            file_meta_path = self.target_files[i]

            filepath = file_meta_path[0]
            file = os.path.basename(filepath)
            subdir = os.path.dirname(filepath)
            meta_filepath = file_meta_path[1]

            with open(meta_filepath, 'r') as f:
                meta = json.load(f)

            bboxes = meta["groundtruth"]["box"]

            if len(bboxes) == 0:
                print("{} has no box".format(file))
                continue

            # reorder boxes
            bboxes_reordered = []
            for box in bboxes:
                min_x = min(box[0], box[2])
                min_y = min(box[1], box[3])
                max_x = max(box[0], box[2])
                max_y = max(box[1], box[3])
                bboxes_reordered.append([min_y, min_x, max_y, max_x])
            bboxes_corrected = np.array(bboxes_reordered)

            image = np.asarray(cv2.imread(filepath), np.int32)
            image = tf.convert_to_tensor(image, np.int32)

            bboxes_normalized = normalize_boxes(bboxes_corrected, image.shape[:2])

            start = time.time()
            (augmented_images, augmented_bboxes) = distort_image_with_autoaugment(image, bboxes_normalized, 'v3')
            end = time.time()
            # print("Took : {}s to augment".format(end-start))
            augmented_bboxes = denormalize_boxes(augmented_bboxes, image.shape[:2])
            augmented_bboxes = tf.cast(augmented_bboxes, dtype=tf.int32)

            start = time.time()
            image_aug = augmented_images.numpy().astype(np.uint8)
            bboxes_aug = augmented_bboxes.numpy()
            end = time.time()
            # print("Took : {}s to retrieve results".format(end-start))
            # print("{}-->{}-->{}".format(bboxes, bboxes_corrected, bboxes_aug))

            bboxes_aug = bboxes_aug.tolist()
            bboxes_aug_corrected = []
            for box in bboxes_aug:
                min_x = min(box[0], box[2])
                min_y = min(box[1], box[3])
                max_x = max(box[0], box[2])
                max_y = max(box[1], box[3])
                bboxes_aug_corrected.append([min_y, min_x, max_y, max_x])

            if self.display:
                # print("Displaying")
                # print(image_aug)
                image_aug_with_box = image_aug.copy()
                # print(image_aug_with_box)
                for box in bboxes_aug_corrected:
                    cv2.rectangle(image_aug_with_box, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                cv2.imshow('augmented', image_aug_with_box)
                k = cv2.waitKey(0)

            # save image inside the database folder with the meta file
            aug_filename = self.set_aug_filename(file, 0)
            folder_name = os.path.basename(subdir)
            folder_augmented = folder_name[:len(folder_name)-3] + "augmented_" + folder_name[len(folder_name)-3:]

            i = 0
            while os.path.isfile(os.path.join(self.output_folder, folder_augmented, aug_filename)):
                i += 1
                # print("File Exists : {}".format(aug_filename))
                aug_filename_ext = os.path.splitext(aug_filename)
                new_aug_filename = aug_filename_ext[0][:len(aug_filename_ext[0])-3]+"{:03d}".format(i)+aug_filename_ext[1]
                aug_filename = new_aug_filename

            # print("Saving to : {}".format(aug_filename))

            aug_image_filepath = os.path.join(self.output_folder, folder_augmented, aug_filename)

            aug_meta_filename = os.path.splitext(aug_filename)[0] + ".meta"
            aug_meta = meta.copy()
            aug_meta["groundtruth"]["box"] = bboxes_aug_corrected
            aug_meta_filepath = os.path.join(self.meta_output_folder, folder_augmented, aug_meta_filename)

            if not os.path.isdir(os.path.dirname(aug_meta_filepath)):
                os.mkdir(os.path.dirname(aug_meta_filepath))
                # print("Created directory : {}".format(os.path.dirname(aug_meta_filepath)))

            with open(aug_meta_filepath, 'w') as f:
                json.dump(aug_meta, f, indent=4, sort_keys=True)
            # print("Wrote : {}".format(aug_meta_filepath))

            if not os.path.isdir(os.path.dirname(aug_image_filepath)):
                os.mkdir(os.path.dirname(aug_image_filepath))
                # print("Created directory : {}".format(os.path.dirname(aug_image_filepath)))

            cv2.imwrite(aug_image_filepath, image_aug)
            # print("Wrote : {}\n".format(aug_image_filepath))

            # if self.display:


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='display_sample', default=False)
    args = parser.parse_args()
    display_sample = args.display_sample

    # input_folder = os.path.join('C:\\', 'Users', 'jonas', 'Desktop', 'Helipad', 'Helipad_DataBase', 'Helipad_DataBase_original')
    # meta_folder = os.path.join('C:\\', 'Users', 'jonas', 'Desktop', 'Helipad', 'Helipad_DataBase_meta', 'Helipad_DataBase_meta_original')
    # root_folder = os.path.join('C:\\', 'Users', 'jonas', 'Desktop', 'Helipad', 'Helipad_DataBase')
    # root_folder_meta = os.path.join('C:\\', 'Users', 'jonas', 'Desktop', 'Helipad', 'Helipad_DataBase_meta')

    input_folder = "../../Helipad_DataBase/Helipad_DataBase_original"
    meta_folder = "../../Helipad_DataBase_meta/Helipad_DataBase_meta_original"
    root_folder = "../../Helipad_DataBase"
    root_folder_meta = "../../Helipad_DataBase_meta"

    balance_dataset = True

    # repartition = None
    #              0  1  2  3  4  5  6  7  8  9  d  u
    repartition = [4, 4, 4, 5, 3, 4, 5, 5, 6, 6, 6, 4]

    database_augmentation = DatabaseAugmentation(input_folder,
                                                 meta_folder,
                                                 root_folder,
                                                 root_folder_meta,
                                                 balance_dataset,
                                                 repartition,
                                                 display_sample)

    database_augmentation.run()

    print("Augmentation Done!")