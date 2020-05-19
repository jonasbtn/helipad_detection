import os
import json
import cv2
import time
import argparse
from tqdm import tqdm
import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage 

import sys
sys.path.append('../')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class DatabaseAugmentationV2:
    
    """
    Apply Augmentation on the dataset using an ImgAug augmentation sequence 
    """
    
    def __init__(self, input_folder, meta_folder, root_folder, root_folder_meta,
                 augmentation_strategy, version_number,
                 balance_dataset=False, repartition=None):
        
        """
        `input_folder`: the folder containing the original images \n
        `meta_folder`: the folder containing the meta of the original images \n
        `root_folder`: the folder where to store the augmented images \n
        `root_folder_meta`: the folder where to store the meta of the augmented images \n
        `augmentation_strategy`: an ImgAug augmentation sequence \n
        `version_number`: the augmentation version number. The suffix of the output folder will be the version number \n 
        `balance_dataset`: boolean, if yes, the number of images per category is balanced by augmenting more the images from small categories \n
        `repartition`: list of 12 integers precising the number of times each image per category is augmented  \n
        """
        
        self.input_folder = input_folder
        self.meta_folder = meta_folder
        self.root_folder = root_folder
        self.root_folder_meta = root_folder_meta
        self.augmentation_strategy = augmentation_strategy
        self.version_number = version_number
        
        self.aug_foldername = self.set_aug_foldername(self.root_folder, version_number)
        self.output_folder = os.path.join(root_folder, self.aug_foldername)
        self.aug_meta_foldername = self.set_aug_foldername(self.root_folder_meta, version_number)
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

    @staticmethod
    def set_aug_foldername(folder, version_number=None):
        """
        Set the augmented folder name with the suffix `version_number`. The output folder is a sub-directory of `folder`. 
        """
        if version_number == None:
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
    
    def set_aug_image_meta_path(self, filename, subdir):
        """
        Save image inside the database folder with the meta file
        """
        aug_filename = self.set_aug_filename(filename, 0)
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
        aug_meta_filepath = os.path.join(self.meta_output_folder, folder_augmented, aug_meta_filename)
        
        return aug_image_filepath, aug_meta_filepath

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
        categories_path = DatabaseAugmentationV2.categories_imagemeta_path(input_folder, meta_folder)
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
    
    @staticmethod
    def load_image_bboxes(image_path, meta_path):
        """
        Load the bounding boxes of the image from `image_path` and `meta_path`\n
        Returns the image, its bounding boxes and its meta informatiosn
        """
        image = cv2.imread(image_path)
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        bboxes = meta["groundtruth"]["box"]
        # reorder boxes
        bboxes_reordered = []
        for box in bboxes:
            min_x = min(box[0], box[2])
            min_y = min(box[1], box[3])
            max_x = max(box[0], box[2])
            max_y = max(box[1], box[3])
            bboxes_reordered.append([min_x, min_y, max_x, max_y])
        return image, bboxes, meta
    
    @staticmethod
    def load_image_box_for_aug(image, bboxes):
        """
        Convert the bounding boxes to ImgAug bounding boxes format \n
        Returns the image with the bounding boxes.
        """
        bb = []
        for box in bboxes:
            bb.append(BoundingBox(x1=box[0], x2=box[2], y1=box[1], y2=box[3]))
        bbs = BoundingBoxesOnImage(bb, shape=image.shape)
        return image, bbs
    
    @staticmethod
    def apply_aug(image, bbs, aug):
        """
        Apply the augmentation on the image using the ImgAug bounding boxes format\n
        Return the augmented image with the augmented bounding boxes
        """
        image_aug, bbs_aug = aug(image=image, bounding_boxes=bbs)
        bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()
        return image_aug, bbs_aug
    
    def apply_augmentation_on_image(self, image, meta, bboxes):
        """
        Apply augmentation on the image using the image and meta files\n
        Return the augmented image with the augmented bounding boxes as a list.
        """
        image, bbs = self.load_image_box_for_aug(image, bboxes)
        image_aug, bbs_aug = self.apply_aug(image, bbs, self.augmentation_strategy)
        bboxes_aug = []
        for bounding_box in bbs_aug.bounding_boxes:
            xmin = int(bounding_box[0][0])
            ymin = int(bounding_box[0][1])
            xmax = int(bounding_box[1][0])
            ymax = int(bounding_box[1][1])
            bboxes_aug.append([xmin, ymin, xmax, ymax])
        return image_aug, bboxes_aug
    
    @staticmethod
    def save_aug(image_aug, bboxes_aug, meta, meta_aug_path, image_aug_path):
        """
        Save the augmented image and meta
        """
        if not os.path.isdir(os.path.dirname(image_aug_path)):
                os.mkdir(os.path.dirname(image_aug_path))
        if not os.path.isdir(os.path.dirname(meta_aug_path)):
                os.mkdir(os.path.dirname(meta_aug_path))
        meta["groundtruth"]["box"] = bboxes_aug
        if "predicted" in meta:
            del meta["predicted"]
        with open(meta_aug_path, 'w') as f:
            json.dump(meta, f, sort_keys=True, indent=4)
        cv2.imwrite(image_aug_path, image_aug)

    def run(self):
        """
        Run the augmentation on the dataset after initialization
        """
        for i in tqdm(range(len(self.target_files))):

            image_meta_path = self.target_files[i]

            image_path = image_meta_path[0]
            filename = os.path.basename(image_path)
            subdir = os.path.dirname(image_path)
            meta_filepath = image_meta_path[1]

            with open(meta_filepath, 'r') as f:
                meta = json.load(f)

            bboxes = meta["groundtruth"]["box"]

            if len(bboxes) == 0:
                print("{} has no box".format(filename))
                continue

            image, bboxes, meta = self.load_image_bboxes(image_path, meta_filepath)
            
            image_aug_path, meta_aug_path = self.set_aug_image_meta_path(filename, subdir)
            
            image_aug, bboxes_aug = self.apply_augmentation_on_image(image,
                                                                     meta,
                                                                     bboxes)
            
            
            aug_meta = meta.copy()
            aug_meta["groundtruth"]["box"] = bboxes_aug
            
            self.save_aug(image_aug, bboxes_aug, meta, meta_aug_path, image_aug_path)


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