import os
import cv2
import matplotlib.pyplot as plt
import shutil
import numpy as np
from tqdm import tqdm as tqdm

np.random.seed(42)


class BBDatasetGroundtruthTMSSplitTestTrain:
    
    """
    Split a bounding boxes dataset separated between "helipad" and "false_positive" into a train and set set for training.\n
    The dataset has to be created first by `BBBuildDataset`. Then, optionnaly, the groundtruth has been verified manually with ``BBDatasetCleaner`.
    """
    
    def __init__(self, image_folder, output_folder, test_size=0.2):
        """
        `image_folder`: path to the image folder\n
        `output_folder`: path to store the dataset splited\n
        `test_size`: float, proportion of image to put inside the test set.
        """
        self.image_folder = image_folder
        self.output_folder = output_folder
        self.test_size = test_size
    
    def build_target_files(self, classe='helipad'):
        """
        Build a list of file path from a specific class\n
        `classe`: the class to load the images\n
        Returns\n
        `target_files`: a list of file path
        """
        target_files = []
        for subdir, dirs, files in os.walk(os.path.join(self.image_folder, classe), topdown=True):
            for file in files:
                target_files.append(os.path.join(subdir, file))
        return target_files
    
    def run(self):
        """
        Run the splitting
        """
        helipad_path = self.build_target_files(classe='helipad')
        false_positive_path = self.build_target_files(classe='false_positive')
        
        l_heli = len(helipad_path)
        l_fp = len(false_positive_path)
        
        nb_heli_test = int(l_heli*(self.test_size))
        index_heli_test = np.random.choice(l_heli, nb_heli_test)
        index_heli_train = [i for i in range(l_heli) if i not in index_heli_test]
        
        nb_fp_test = int(l_fp*(self.test_size))
        index_fp_test = np.random.choice(l_fp, nb_fp_test)
        index_fp_train = [i for i in range(l_fp) if i not in index_fp_test]
        
        helipad_train_path = np.array(helipad_path)[index_heli_train]
        helipad_test_path = np.array(helipad_path)[index_heli_test]
        
        fp_train_path = np.array(false_positive_path)[index_fp_train]
        fp_test_path = np.array(false_positive_path)[index_fp_test]
        
        # copy file to train/helipad train/false_positive
        # copy file to test/helipad test/false_positive
        
        if not os.path.isdir(os.path.join(output_folder, 'train')):
            os.mkdir(os.path.join(output_folder, 'train'))
        if not os.path.isdir(os.path.join(output_folder, 'test')):
            os.mkdir(os.path.join(output_folder, 'test'))
        if not os.path.isdir(os.path.join(output_folder, 'train', 'helipad')):
            os.mkdir(os.path.join(output_folder, 'train', 'helipad'))
        if not os.path.isdir(os.path.join(output_folder, 'train', 'false_positive')):
            os.mkdir(os.path.join(output_folder, 'train', 'false_positive'))
        if not os.path.isdir(os.path.join(output_folder, 'test', 'helipad')):
            os.mkdir(os.path.join(output_folder, 'test', 'helipad'))
        if not os.path.isdir(os.path.join(output_folder, 'test', 'false_positive')):
            os.mkdir(os.path.join(output_folder, 'test', 'false_positive'))
        
        for path in helipad_train_path:
            shutil.copy(path, os.path.join(output_folder, 'train', 'helipad', os.path.basename(path)))
        for path in helipad_test_path:
            shutil.copy(path, os.path.join(output_folder, 'test', 'helipad', os.path.basename(path)))
        for path in fp_train_path:
            shutil.copy(path, os.path.join(output_folder, 'train', 'false_positive', os.path.basename(path)))
        for path in fp_test_path:
            shutil.copy(path, os.path.join(output_folder, 'test', 'false_positive', os.path.basename(path)))
        
        print('Done')
        

if __name__ == "__main__":
    image_folder = "C:\\Users\AISG\\Documents\\Jonas\\Helipad\\Real_World_Detected_Boxes\\model_10_0.0_groundtruth\\"
    output_folder = "C:\\Users\AISG\\Documents\\Jonas\\Helipad\\Real_World_Detected_Boxes\\model_10_0.0_groundtruth_split\\"
    test_size = 0.2

    bb_dataset_groundtruth_tms_split_test_train = BBDatasetGroundtruthTMSSplitTestTrain(image_folder=image_folder,
                                                                                       output_folder=output_folder,
                                                                                       test_size=test_size)

    bb_dataset_groundtruth_tms_split_test_train.run()