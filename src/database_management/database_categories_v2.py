import os
import matplotlib.pyplot as plt
import cv2
import shutil
import numpy as np
import json


class DatabaseCategoriesv2:

    """
    Interface to assign an helipad to a category. The dataset has to be manually annotated first. This second version saves the category directly inside the meta files. The images are not copied. 
    """
    def __init__(self, image_folder_original, meta_folder_original, nb_categories=12):
        """
        `image_folder_original`: folder containing the original image dataset \n
        `meta_folder_original`: folder containing the original meta files.
        `nb_categories`: the number of categories wanted. 
        """
        self.image_folder = image_folder_original
        self.meta_folder = meta_folder_original
        self.nb_categories = nb_categories

        self.files_per_categories = []
        for i in range(nb_categories):
            self.files_per_categories.append([])

        self.target_files = self.build_target_list()
        self.load_last_image_per_category()

    def convert_cat_str_to_int(self, str_cat):
        """
        Convert a string into an int. \n
        `str_cat`: the category as string \n
        Returns an int
        """
        for i in range(10):
            if str_cat == str(i):
                return i
        if str_cat == "d":
            return 10
        elif str_cat == "u":
            return 11

    def build_target_list(self):
        """
        Build a list of target files. Each target file is a tuple paths ('imagepath', 'metapath') to load the files.\n
        Return a list of tuples. 
        """
        target_files = []
        for subdir, dirs, files in os.walk(self.image_folder, topdown=True):
            for file in files:
                imagepath = os.path.join(subdir, file)
                metapath = os.path.join(self.meta_folder,
                                         os.path.basename(subdir),
                                         os.path.splitext(file)[0]+".meta")
                with open(metapath, 'r') as f:
                    meta = json.load(f)
                if not "groundtruth" in meta:
                    continue
                elif not meta["groundtruth"]["helipad"]:
                    continue
                elif "box" not in meta["groundtruth"]:
                    continue
                elif "category" in meta["groundtruth"]:
                    str_cat = meta["groundtruth"]["category"]
                    int_cat = self.convert_cat_str_to_int(str_cat)
                    self.files_per_categories[int_cat].append([imagepath, metapath])
                    continue
                target_files.append([imagepath, metapath])
        return target_files

    def load_last_image_per_category(self):
        """
        Load the last image of each category and group them into one image for display.
        """
        self.nb_row = 4
        self.nb_col = 3
        self.img_width = 640
        self.img_height = 640
        self.final_img = np.zeros((self.img_width*self.nb_row, self.img_height*self.nb_col, 3))
        i = 0
        j = 0

        for i in range(len(self.files_per_categories)):
            nb_cat = i
            files_cat = self.files_per_categories[nb_cat]
            if len(files_cat) == 0:
                continue
            last_image_meta_path = files_cat[len(files_cat)-1]
            last_image_path = last_image_meta_path[0]
            last_meta_path = last_image_meta_path[1]

            img = cv2.imread(last_image_path)
            img = img/256
            # img_r = cv2.resize(img.copy(), (self.img_width, self.img_height))
            # print(img_r.shape)
            i = nb_cat // self.nb_col
            j = nb_cat % self.nb_col

            self.final_img[i * self.img_width:(i+1) * self.img_width, j * self.img_height:(j + 1) * self.img_height, :] = img

        self.final_img_r = cv2.resize(self.final_img, (640, 640))

    def add_to_category(self, image_path, meta_path, cat_nb):
        """
        Add an `image_path` and `meta_path` to a category `cat_nb`
        """
        self.files_per_categories[cat_nb].append([image_path, meta_path])

    def reload_grid_category(self, img, nb_cat):
        """
        Reload the grid of the last image per category.\n
        `nb_cat`: the category number\n
        `img`: the image to add to the grid\n
        Display the grid of images
        """
        img = img/256
        i = nb_cat // self.nb_col
        j = nb_cat % self.nb_col
        img_r = cv2.resize(img.copy(), (self.img_width, self.img_height))
        self.final_img[i * self.img_width:(i + 1) * self.img_width, j * self.img_height:(j + 1) * self.img_height, :] = img_r
        self.final_img_r = cv2.resize(self.final_img, (640,640))
        cv2.imshow('Category', self.final_img_r)

    def build_categories(self):
        """
        Run the interface after initialization.
        """
        i = 0

        window_name = 'image'

        while i < len(self.target_files):

            image_meta_path = self.target_files[i]
            image_path = image_meta_path[0]
            meta_path = image_meta_path[1]
            print(image_path)

            img = cv2.imread(image_path)
            cv2.imshow(window_name, img)

            cv2.imshow('Category', self.final_img_r)

            k = cv2.waitKey(0)

            if k >= 48 and k <= 57:
                target = str(k % 48)
            elif k == 100:
                target = 'd'
            elif k == 117:
                target = 'u'
            elif k == 8 : #back
                target = -2
            else:
                target = -1

            print("target = {}".format(target))

            if target is -1:
                i += 1
                continue
            elif target is -2:
                # delete
                i = i-1
                # find last image path and meta path inside the category
                image_meta_path = self.target_files[i]
                for j in range(self.nb_categories):
                    if image_meta_path in self.files_per_categories[j]:
                        meta_path = image_meta_path[1]
                        # open the meta file and remove the category key
                        with open(meta_path, 'r') as f:
                            meta = json.load(f)
                        groundtruth = meta["groundtruth"]
                        del groundtruth["category"]
                        meta["groundtruth"] = groundtruth
                        with open(meta_path, 'w') as f:
                            json.dump(meta, f, indent=4, sort_keys=True)
                        self.files_per_categories[j].remove(image_meta_path)
                        print("File removed from category!")
                        break

                # Redisplay the previous category image inside the grid
                last_cat_img_meta_path = self.files_per_categories[int_cat][len(self.files_per_categories[int_cat])-1]
                image_path = last_cat_img_meta_path[0]
                meta_path = last_cat_img_meta_path[1]
                img = cv2.imread(image_path)
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                str_cat = meta["groundtruth"]["category"]
                int_cat = self.convert_cat_str_to_int(str_cat)
                self.reload_grid_category(img, int_cat)
                continue
            else:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                meta["groundtruth"]["category"] = target
                with open(meta_path, 'w') as f:
                    json.dump(meta, f, indent=4, sort_keys=True)
                int_cat = self.convert_cat_str_to_int(target)
                self.files_per_categories[int_cat].append(image_meta_path)

                self.reload_grid_category(img, int_cat)

            i = i + 1

            print("{} more to go!".format(len(self.target_files)-i))


if __name__ == "__main__":

    image_folder = os.path.join('C:\\', 'Users', 'jonas', 'Desktop', 'Helipad', 'Helipad_DataBase', 'Helipad_DataBase_original')
    meta_folder = os.path.join('C:\\', 'Users', 'jonas', 'Desktop', 'Helipad', 'Helipad_DataBase_meta', 'Helipad_DataBase_meta_original')

    database_categories = DatabaseCategoriesv2(image_folder, meta_folder, nb_categories=12)

    database_categories.build_categories()





