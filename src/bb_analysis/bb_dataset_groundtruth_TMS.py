import os
import cv2
import matplotlib.pyplot as plt
from IPython.display import clear_output
import shutil
import numpy as np
from tqdm import tqdm as tqdm


class BBDatasetGroundtruthTMS:
    
    """
    Build the groundtruth for bounding boxes coming from a TMS Dataset.\n
    The dataset has to be first created by `BBBuildDataset`\n
    For each image, press `y` for an helipad or `n` for a false positive. The bounding box is then moved to the other classe. Press `p` to go back to the previous image.\n
    If the groundtruth from the detection made by the same model has been done before for another `zoom_out`, the same groundtruth can be apply directly using the previous one. Specify the folder of the already made groundtruth in `source_from` to automatically rebuild the groundtruth.\n
    """
    
    def __init__(self, image_folder, output_folder, start_index=0, source_from=None):
        """
        'image folder' contains 2 folders : 'helipad' and 'false_positive'
        `output_folder`: string, path to where to dataset has to be stored\n
        `start_index`: int, index to where to start the verification.
        `source_from`: string, path to a previously made groundtruth for the same model
        """
        self.image_folder = image_folder
        self.output_folder = output_folder
        self.start_index = start_index
        self.source_from = source_from
        
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
        if not os.path.isdir(os.path.join(output_folder, 'helipad')):
            os.mkdir(os.path.join(output_folder, 'helipad'))
        if not os.path.isdir(os.path.join(output_folder, 'false_positive')):
            os.mkdir(os.path.join(output_folder, 'false_positive'))
    
    def build_target_files(self):
        """
        Build the list of target files
        """
        target_files = []
        for subdir, dirs, files in os.walk(self.image_folder, topdown=True):
            for file in files:
                target_files.append(os.path.join(subdir, file))
        return target_files
    
    def build_groundtruth_from_other_folder(self, target_files):
        """
        Build the groundtruth from a previously made groundtruth\n
        `target_files`: list, the list of target files\n
        """
        helipad_filenames = os.listdir(os.path.join(self.source_from, 'helipad'))
        fp_filenames = os.listdir(os.path.join(self.source_from, 'false_positive'))
        for i in tqdm(range(len(target_files))):
            filepath = target_files[i]
            filename = os.path.basename(filepath)
            if filename in helipad_filenames:
                # yes, move image from input to output
                shutil.copy(filepath,
                            os.path.join(self.output_folder, 
                                        'helipad',
                                         filename))
            elif filename in fp_filenames:
                shutil.copy(filepath,
                            os.path.join(self.output_folder, 
                                        'false_positive',
                                         filename))
            else:
                print('Groundtruth not found for {}'.format(filename))

    def run(self):
        """
        Run the interface
        """
        target_files = self.build_target_files()
        print(f'{len(target_files)} files loaded!')
        
        if self.source_from:
            self.build_groundtruth_from_other_folder(target_files)
        else:
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
                    # yes, move image from input to output
                    shutil.copy(filepath,
                                os.path.join(self.output_folder, 
                                            'helipad',
                                            os.path.basename(filepath)))
                elif key == 'n':
                    shutil.copy(filepath,
                               os.path.join(self.output_folder, 
                                            'false_positive',
                                            os.path.basename(filepath)))

                i = i + 1
                clear_output()
                

if __name__ == "__main__":
    image_folder = "C:\\Users\AISG\\Documents\\Jonas\\Helipad\\Real_World_Detected_Boxes\\model_10_0.0_zoomout5\\sat"
    output_folder = "C:\\Users\AISG\\Documents\\Jonas\\Helipad\\Real_World_Detected_Boxes\\model_10_0.0_zoomout5_groundtruth\\"
    source_folder = "C:\\Users\AISG\\Documents\\Jonas\\Helipad\\Real_World_Detected_Boxes\\model_10_0.0_groundtruth\\"
    start_index = 0

    bb_build_groundtruth_tms = BBDatasetGroundtruthTMS(image_folder=image_folder,
                                                       output_folder=output_folder,
                                                       start_index=0,
                                                       source_from=source_folder)

    bb_build_groundtruth_tms.run()