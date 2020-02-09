import os
import matplotlib.pyplot as plt
import cv2
import shutil


class DatabaseCleaner:

    def __init__(self, folder, true_folder, false_folder, unknown_folder, mode=1):

        if not os.path.isdir(true_folder):
            os.mkdir(true_folder)
        if not os.path.isdir(false_folder):
            os.mkdir(false_folder)
        if not os.path.isdir(unknown_folder):
            os.mkdir(unknown_folder)

        self.folder = folder
        self.true_folder = true_folder
        self.false_folder = false_folder
        self.unknown_folder = unknown_folder
        self.mode = mode

        self.all_filepaths = self.load_all_filepaths(folder)
        self.true_filepath = self.load_all_filepaths(true_folder)
        self.false_filepath = self.load_all_filepaths(false_folder)
        self.unknown_filepath = self.load_all_filepaths(unknown_folder)

        print(self.all_filepaths)
        print(len(self.all_filepaths))

    def load_all_filepaths(self, folder):
        all_filepaths = []
        for subdir, dirs, files in os.walk(folder):
            for file in files:
                filepath = os.path.join(subdir, file)
                all_filepaths.append(filepath)
        return all_filepaths

    @staticmethod
    def copy_file(filename, source, directory, destination_database):
        destination_folder = os.path.join(destination_database, directory)
        if not os.path.isdir(destination_folder):
            os.mkdir(destination_folder)
        destination_file = os.path.join(destination_folder, filename)
        dest = shutil.copy(source, destination_file)
        print("Copied to : %s" % destination_file)

    def groundtruth(self):

        i = 0
        window_name = 'image'

        while i < len(self.all_filepaths):

            source = self.all_filepaths[i]
            filename = os.path.basename(source)
            directory = os.path.basename(os.path.dirname(source))

            # already
            if os.path.join(self.true_folder, directory, filename) in self.true_filepath \
                    or os.path.join(self.false_folder, directory, filename) in self.false_filepath \
                    or os.path.join(self.unknown_folder, directory, filename) in self.unknown_filepath:
                print("image {} already classified".format(filename))
                i = i+1
                continue
            # pass

            image = cv2.imread(source)
            cv2.imshow(window_name, image)
            k = cv2.waitKey(0)

            # yes
            if k == 121:
                self.copy_file(filename, source, directory, self.true_folder)
                self.true_filepath.append(os.path.join(self.true_folder, directory, filename))
            # no
            elif k == 110:
                self.copy_file(filename, source, directory, self.false_folder)
                self.false_filepath.append(os.path.join(self.false_folder, directory, filename))
            # unknown
            elif k == 117:
                self.copy_file(filename, source, directory, self.unknown_folder)
                self.unknown_filepath.append(os.path.join(self.unknown_folder, directory, filename))
            # back
            elif k == 8:
                i = i-1
                # Add Delete
                source = self.all_filepaths[i]
                filename = os.path.basename(source)
                directory = os.path.basename(os.path.dirname(source))

                true_path = os.path.join(self.true_folder, directory, filename)
                false_path = os.path.join(self.false_folder, directory, filename)
                unknown_path = os.path.join(self.unknown_folder, directory, filename)
                if os.path.exists(true_path):
                    os.remove(true_path)
                    self.true_filepath.remove(true_path)
                    print("File %s removed!" % true_path)
                if os.path.exists(false_path):
                    os.remove(false_path)
                    self.false_filepath.remove(false_path)
                    print("File %s removed!" % false_path)
                if os.path.exists(unknown_path):
                    os.remove(unknown_path)
                    self.unknown_filepath.remove(unknown_path)
                    print("File %s removed!" % unknown_path)
                continue

            print(k)

            i = i+1

            print("{} more to go!".format(len(self.all_filepaths)-i))

    def run(self):

        if self.mode == 1:
            self.groundtruth()


if __name__ == "__main__":
    folder = os.path.join('C:\\', 'Users', 'jonas', 'Desktop', 'Helipad_DataBase')

    true_folder = os.path.join('C:\\', 'Users', 'jonas', 'Desktop', 'Helipad_DataBase_True')
    false_folder = os.path.join('C:\\', 'Users', 'jonas', 'Desktop', 'Helipad_DataBase_False')
    unknown_folder = os.path.join('C:\\', 'Users', 'jonas', 'Desktop', 'Helipad_DataBase_Junk')

    database_cleaner = DatabaseCleaner(folder, true_folder, false_folder, unknown_folder, mode=1)

    database_cleaner.run()
