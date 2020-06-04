import os
import json
import cv2
import matplotlib.pyplot as plt
from IPython.display import clear_output


class BuildGroundtruthTMS:
    
    """
    Build the groundtruth by annotating `True` or `False` on bounding boxes from satellite images that were previously predicted by a model.\n
    Save the manual validation inside the meta files under the key:\n
    `meta["predicted"][f"model_{model_number}"]["groundtruth"] = [True/False]`\n
    If a dataset of bounding boxes has already been created inside a folder using `BBBuildDataset` and annotated using `BBDatasetGroundtruthTMS`, it is possible to automatically save the annotation inside the meta file by specifying such a folder. 
    """
    
    def __init__(self, image_folder, meta_folder, model_number, index_path=None, source_from=None, start_index=0):
        """
        'image folder' contains 2 folders : 'helipad' and 'false_positive'
        `meta_folder`: string, path to the meta folder\n
        `model_number`: int, number of the model that predicted to bounding boxes\n
        `index_path`: string or None, path to the index files containing the names of the files with a bounding box inside\n
        `source_from`: string, path to a previously made groundtruth for the same model\n
        `start_index`: int, index to where to start the verification.\n
        """
        
        self.image_folder = image_folder
        self.meta_folder = meta_folder
        self.model_number = model_number
        self.index_path = index_path
        self.source_from = source_from
        self.start_index = start_index
        self.tms = True
        if self.index_path and self.tms:
            self.target_files = self.convert_meta_filename_to_path(self.image_folder, self.meta_folder, self.index_path)
        else:
            self.target_files = self.load_target_files()
    
    def load_target_files(self):
        """
        Load the target files.\n
        Returns a list of tuple (`image_path`, `meta_path`) 
        """
        target_files = []
        for subdir, dirs, files in os.walk(self.image_folder, topdown=True):
            for file in files:
                image_path = os.path.join(subdir, file)
                if self.tms:
                    ytile = os.path.splitext(file)[0]
                    xtile = os.path.basename(subdir)
                    zoom = os.path.basename(os.path.dirname(subdir))
                    meta_path = os.path.join(self.meta_folder,
                                             zoom,
                                             xtile,
                                             "Satellite_{}_{}_{}.meta".format(zoom,
                                                                              xtile,
                                                                              ytile))
                else:
                    folder_id = os.path.basename(subdir)
                    filename = os.path.splitext(file)[0]
                    meta_path = os.path.join(self.meta_folder,
                                             self.folder_id,
                                             filename+'.meta')
                target_files.append([image_path, meta_path])
        return target_files
    
    @staticmethod
    def convert_meta_filename_to_path(image_folder, meta_folder, index_path):
        """
        From the index file, convert each  meta filename to a tuple (`image_path`, `meta_path`)\n
        `image_folder`: string, path to the root of the image folder\n
        `meta_folder`: string, path to the root of the meta_folder\n
        `index_path`: string, path to the index file
        """
        image_meta_path = []
        with open(index_path, 'r') as f:
            for meta_filename in f:
                info = meta_filename.split('_')
                ytile = info[3].split('.')[0]
                xtile = info[2]
                zoom = info[1]
                meta_path = os.path.join(meta_folder,
                                         zoom,
                                         xtile,
                                         "Satellite_{}_{}_{}.meta".format(zoom,
                                                                          xtile,
                                                                          ytile))
                image_path = os.path.join(image_folder,
                                          zoom,
                                          xtile, 
                                          ytile+".jpg")
                image_meta_path.append([image_path, meta_path])
        f.close()
        return image_meta_path
    
    def build_image_groundtruth_from_other_folder(self, meta_filename, box_id):
        """
        Build the groundtruth from a previously made groundtruth\n
        `meta_filename`: string, the basename of the image\n
        `box_id`: int, the id of the bounding box\n
        """
        
        filename = os.path.splitext(meta_filename)[0]
        bb_filename = filename + f'_{box_id}.jpg' 
        
        helipad_filenames = os.listdir(os.path.join(self.source_from, 'helipad'))
        fp_filenames = os.listdir(os.path.join(self.source_from, 'false_positive'))
        
        if bb_filename in helipad_filenames:
            return True
        elif bb_filename in fp_filenames:
            return False
        else:
            return False
        
    def run(self):
        """
        Run the interface
        """
        print(f'{len(self.target_files)} files loaded!')
        
        l = len(self.target_files)
        i = self.start_index
        
        while i<l:
            
            image_meta_path = self.target_files[i]
            image_path = image_meta_path[0]
            meta_path = image_meta_path[1]
            
            image = cv2.imread(image_path)
            
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            f.close()
            
            if "predicted" not in meta:
                continue
            elif f'model_{self.model_number}' not in meta["predicted"]:
                continue
            elif len(meta["predicted"][f'model_{self.model_number}']["box"]) == 0:
                continue
            
            bboxes = meta["predicted"][f'model_{self.model_number}']["box"]
            bboxes_groundtruth = []
            
            for j in range(len(bboxes)):
                
                if self.source_from:
                    # get annotation from the source from folder with box id j
                    bb_groundtruth = self.build_image_groundtruth_from_other_folder(os.path.basename(meta_path), j)
                    
                    if bb_groundtruth is None:
                        continue
                    bboxes_groundtruth.append(bb_groundtruth)
                    continue
                    
                box = bboxes[j]
                
                x_min = min(box[0], box[2])
                y_min = min(box[1], box[3])
                x_max = max(box[0], box[2])
                y_max = max(box[1], box[3])
                
                image_box = image[y_min:y_max,x_min:x_max,:]
                
                plt.imshow(image_box)
                plt.show()

                key = input()

                while key != 'y' and key != 'n' and key != 'p':
                    key = input()

                print(key)

                if key == 'p':
                    i = i-2
                    continue
                if key == 'y':
                    bboxes_groundtruth.append(True)
                elif key == 'n':
                    bboxes_groundtruth.append(False)

                clear_output()

            meta["predicted"][f'model_{self.model_number}']["groundtruth"] = bboxes_groundtruth
            
            with open(meta_path, 'w') as f:
                json.dump(meta, f, sort_keys=True, indent=4)
            f.close()
            
            i += 1


if __name__ == "__main__":
    
    image_folder = "C:\\Users\\AISG\\Documents\\Jonas\\Real_World_Dataset_TMS\\sat\\"
    meta_folder = "C:\\Users\\AISG\\Documents\\Jonas\\Real_World_Dataset_TMS_meta\\sat\\"
    model_number = 10
    index_path = "C:\\Users\\AISG\\Documents\\Jonas\\helipad_detection\\src\\helipad_path_over_0_m10.txt"
    source_from = "C:\\Users\\AISG\\Documents\\Jonas\\Helipad\\Real_World_Detected_Boxes\\model_10_0.0_groundtruth\\"
    start_index = 0

    build_groundtruth_tms = BuildGroundtruthTMS(image_folder=image_folder,
                                                meta_folder=meta_folder,
                                                model_number=model_number,
                                                index_path=index_path,
                                                source_from=source_from,
                                                start_index=start_index)

    build_groundtruth_tms.run()
                    