import os
import matplotlib.pyplot as plt
import cv2
import shutil
import numpy as np


class DatabaseCategories:
    
    """
    Interface to assign an helipad to a category. This version creates a new folder into which the images are copied from the dataset folder into subdirectories named after the category chosen.
    """

    def __init__(self, true_folder, category_folder, nb_categories=12):
        """
        `true_folder`: the folder containing the helipad images\n
        `category_folder`: the output root folder when to copy the helipad image. A subdirectory will be created for each category.\n
        `nb_categories`: the number of categories wanted.
        """

        if not os.path.isdir(true_folder):
            os.mkdir(true_folder)
        if not os.path.isdir(category_folder):
            os.mkdir(category_folder)
        for i in range(nb_categories-2):
            if not os.path.isdir(os.path.join(category_folder, "Helipad_category_{}".format(i))):
                os.mkdir(os.path.join(category_folder, "Helipad_category_{}".format(i)))
        if not os.path.isdir(os.path.join(category_folder, "Helipad_category_d")):
            os.mkdir(os.path.join(category_folder, "Helipad_category_d"))
        if not os.path.isdir(os.path.join(category_folder, "Helipad_category_u")):
            os.mkdir(os.path.join(category_folder, "Helipad_category_u"))

        self.category_folder_basename = "Helipad_category_"

        self.true_folder = true_folder
        self.category_folder = category_folder
        self.nb_categories = nb_categories

        self.all_filepaths = self.load_all_filepaths(true_folder)
        self.categories_file = self.load_categories_file(category_folder)
        print(self.categories_file)
        self.nb_blank = nb_categories
        self.example_img = self.load_first_image_per_category(category_folder)

        print(len(self.all_filepaths))

    def load_all_filepaths(self, folder):
        """
        Load all the filepaths inside a folder.\n
        Returns a list containing all the filepaths.
        """
        all_filepaths = []
        for subdir, dirs, files in os.walk(folder):
            for file in files:
                filepath = os.path.join(subdir, file)
                all_filepaths.append(filepath)
        return all_filepaths

    def load_categories_file(self, folder):
        """"
        `folder` is the root folder containing the subfolders of all the categories.\n
        Return a dictionnary containing the categories as a key and the list of filepaths as values.
        """
        categories_file = {}
        for root, dirs, files in os.walk(folder, topdown=True):
            folder = os.path.basename(root)
            if len(folder.split('_')) == 2:
                category = os.path.basename(os.path.dirname(root))
                if category[len(category)-1:] is 's':
                    continue
                category_number = category[len(category)-1:]
                if len(files) == 0:
                    if category_number not in categories_file:
                        categories_file[category_number] = []
                else:
                    for file in files:
                        if category_number not in categories_file:
                            categories_file[category_number] = [file]
                        else:
                            categories_file[category_number].append(file)
        return categories_file

    def add_file_to_category(self, target, file):
        """
        Add a `file` path to a category `target`\n
        `file` is a path. 
        
        """
        if target not in self.categories_file:
            self.categories_file[target] = [file]
        else:
            self.categories_file[target].append(file)

    def load_first_image_per_category(self, folder):
        """
        Load the first image per category\n
        Returns an image containing a grid of images with each image belonging to a different category.
        """
        img_width = 640
        img_height = 640
        nb_row = 4
        nb_col = 3
        final_img = np.zeros((img_width*nb_row, img_height*nb_col, 3))
        i = 0
        j = 0
        for root, dirs, files in os.walk(folder, topdown=True):
            folder = os.path.basename(root)
            if len(folder.split('_')) == 2:
                # Folder_001
                category = os.path.basename(os.path.dirname(root))
                target = category[len(category)-1:]
                if target is 's':
                    continue
                if len(files) == 0:
                    img = np.zeros((img_width, img_height, 3))
                else:
                    filepath = os.path.join(root, files[len(files)-1])
                    img = cv2.imread(filepath)
                    # normalize
                    img = img/256
                    self.nb_blank -= 1
                if target is 'd':
                    i = 3
                    j = 1
                elif target is 'u':
                    i = 3
                    j = 2
                else:
                    nb_category = int(target)
                    i = nb_category // nb_col
                    j = nb_category % nb_col

                final_img[i * img_width:(i + 1) * img_width, j * img_height:(j + 1) * img_height, :] = img

        final_img = cv2.resize(final_img, (640, 640))
        return final_img

    def add_to_category(self, category_number, filename):
        """
        Add a `filename` to `category_number` inside the dictionnary `self.categories_file`.
        """
        if category_number not in self.categories_file:
            self.categories_file[category_number] = [filename]
        else:
            self.categories_file[category_number].append(filename)

    def load_filepath_first_image_categories(self):
        """
        Load the filepath of the first image of each category.\n
        Returns a list of filepath.
        """
        filepaths = []
        for category_number in self.categories_file.keys():
            folder = "Helipad_category_{:02d}".format(category_number)
            filename = self.categories_file[category_number][0]
            filepath = os.path.join(self.category_folder, folder, filename)
            filepaths.append(filepath)
        return filepaths

    def display_categories(self):
        """
        Display all the categories
        """
        self.example_img = self.load_first_image_per_category(self.category_folder)
        cv2.imshow('Category', self.example_img)

    @staticmethod
    def copy_file(filename, source, directory, destination_database):
        """
        Static method to copy a file `filename` from `source` to `os.path.join(destination_database, directory)`
        """
        destination_folder = os.path.join(destination_database, directory)
        if not os.path.isdir(destination_folder):
            os.mkdir(destination_folder)
        destination_file = os.path.join(destination_folder, filename)
        dest = shutil.copy(source, destination_file)
        print("Copied to : %s" % destination_file)

    def build_categories(self):
        """
        Launch the interface to annotate the categories.
        """
        i = 0
        window_name = 'image'

        while i < len(self.all_filepaths):

            source = self.all_filepaths[i]
            filename = os.path.basename(source)
            directory = os.path.basename(os.path.dirname(source))

            already_classified = False
            for categorie_number, filenames in self.categories_file.items():
                if not already_classified:
                    if filename in filenames:
                        already_classified = True
            if already_classified:
                print("image {} already classified".format(filename))
                i = i+1
                continue

            image = cv2.imread(source)
            cv2.imshow(window_name, image)

            self.display_categories()

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
                source = self.all_filepaths[i]
                filename = os.path.basename(source)
                directory = os.path.basename(os.path.dirname(source))

                for category, files in self.categories_file.items():
                    if filename in files:
                        path = os.path.join(self.category_folder, self.category_folder_basename+category, directory, filename)
                        if os.path.exists(path):
                            os.remove(path)
                            files.remove(filename)
                            print("File %s removed!" % path)
                continue
            else:
                destination = os.path.join(self.category_folder, self.category_folder_basename+target)
                self.copy_file(filename, source, directory, destination)
                self.add_file_to_category(target, filename)
                # self.load_categories_file(self.category_folder)

            i = i + 1

            print("{} more to go!".format(len(self.all_filepaths) - i))

    def run(self):
        """
        Run the interface
        """
        self.build_categories()


if __name__ == "__main__":

    true_folder = os.path.join('C:\\', 'Users', 'jonas', 'Desktop', 'Helipad', 'Helipad_DataBase_True')
    category_folder = os.path.join('C:\\', 'Users', 'jonas', 'Desktop', 'Helipad', 'Helipad_DataBase_Categories')

    database_category = DatabaseCategories(true_folder, category_folder)

    database_category.run()

