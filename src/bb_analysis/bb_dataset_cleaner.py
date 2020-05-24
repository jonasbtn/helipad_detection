import os
import cv2
import matplotlib.pyplot as plt
from IPython.display import clear_output
import shutil


class BBDatasetCleaner:
    
    """
    Interface to manually verify the bounding boxes inside a classe.\n
    The dataset has to be created first by `BBBuildDataset`.\n
    For each image, press `y` for confirm the good classification or `n` to confirm the false classification. The bounding box is then moved to the other classe. Press `p` to go back to the previous image.\n
    """
    
    def __init__(self, image_folder, check_false_positive=True, start_index=0):
        """
        'image folder' contains 2 folders : 'helipad' and 'false_positive'\n
        `check_false_positive`: boolean, True to verify the false positives, False to verify the helipads\n
        `start_index`: int, index to where to start the verification.
        """
        self.image_folder = image_folder
        self.check_false_positive = check_false_positive
        self.start_index = start_index
        
        if check_false_positive:
            self.input_folder = os.path.join(self.image_folder,
                                               'false_positive')
            self.output_folder = os.path.join(self.image_folder,
                                             'helipad')
        else:
            self.input_folder = os.path.join(self.image_folder,
                                               'helipad')
            self.output_folder = os.path.join(self.image_folder,
                                             'false_positive')
    
    def build_target_files(self):
        """
        Build the list of target files 
        """
        target_files = []
        for subdir, dirs, files in os.walk(self.input_folder, topdown=True):
            for file in files:
                target_files.append(os.path.join(subdir, file))
        return target_files
    
    def run(self):
        """
        Run the interface
        """
        target_files = self.build_target_files()
        l = len(target_files)
        nb_move = 0
        i = self.start_index
        
        while i < l:
            print(i)
            print(f'{l-i} files remaining!')
            
            filepath = target_files[i]
            print(filepath)
            
            image = plt.imread(filepath)

            plt.imshow(image)
            plt.show()

            key = input()

            while key != 'y' and key != 'n' and key != 'p':
                key = input()

            print(key)
            
            if key == 'p':
                i = i-1
                continue
            if key == 'y':
                if not os.path.isfile(os.path.join(self.output_folder, os.path.basename(filepath))):                              
                    # yes, move image from input to output
                    shutil.move(filepath,
                               self.output_folder, os.path.basename(filepath))  
                    print('Image moved')
                else:
                    os.remove(os.path.join(self.input_folder, os.path.basename(filepath)))
                    print('Image deleted')
            
            i = i + 1
            clear_output()
            

if __name__ == "__main__":
    image_folder = "C:\\Users\AISG\\Documents\\Jonas\\Detected_Boxes_3\\model_7_0.0\\test\\"
    check_false_positive = True
    start_index = 0

    bb_dataset_cleaner = BBDatasetCleaner(image_folder=image_folder,
                                         check_false_positive=check_false_positive,
                                         start_index=start_index)

    bb_dataset_cleaner.run()