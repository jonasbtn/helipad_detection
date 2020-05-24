import os
import json
from tqdm import tqdm as tqdm
import cv2

from shadow_detection import ShadowDetection


class RunShadowDetection:
    
    """
    Run the shadow detection on image inside an image folder and saves the results inside the according meta file under the following key:\n
    `meta["groundtruth"]["shadow"]['zoom_out_{}'.format(self.zoom_out)]`\n
    or\n
    `meta["predicted"]["model_{}".format(self.model_number)]["shadow"]['zoom_out_{}'.format(self.zoom_out)]`\n
    """
    
    def __init__(self, image_folder, meta_folder, model_number, groundtruth_only=False, tms=True, zoom_out=0, index_path=None,
                minimum_size_window=3, threshold_v=0.35, threshold_s=0.02, ratio=1, d_0=3):
        """
        `image_folder`: string, path to the image folder\n
        `meta_folder`: string, path to the meta folder\n
        `model_number`: int, number of the model that predicted to bounding boxes\n
        `groundtruth_only`: boolean, True to run the shadow detection on the groundtruth bounding boxes only\n
        `tms`: boolean, True if the `image_folder` follows the TMS's directory structure\n
        `zoom_out`: int, increase size of the bounding boxes in pixels in every direction\n
        `index_path`: string or None, path to the index files containing the names of the files with a bounding box inside\n
        `minimum_size_window`: the minimum size of a shadow is defined by a square of side `(minimum_size_window*2-1)`\n
        `threshold_v`: the mean of the window in V must be inferior than `threshold_v` to be accepted.\n
        `threshold_s`: the mean of the window in S must be superior than `threshold_s` to be accepted.\n
        `ratio`: all the values of the window in `c3` must be superior than `mean(c3)*ratio`.\n
        `d_0`: the candidate pixel to be added to the region shadow must be below a Mahalanobis distance `d_0` from the `mean(c3[region])`.
        """
        self.image_folder = image_folder
        self.meta_folder = meta_folder
        self.model_number = model_number
        self.groundtruth_only = groundtruth_only
        self.tms = tms
        self.zoom_out = zoom_out
        self.index_path = index_path
        self.minimum_size_window = minimum_size_window
        self.threshold_v = threshold_v
        self.threshold_s = threshold_s
        self.ratio = ratio
        self.d_0 = d_0
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
    
    @staticmethod
    def box_zoom_out(image, x_min, y_min, x_max, y_max, zoom_out):
        """
        Increase the size of the bounding box by `zoom_out` pixels.\n
        Returns \n
        `image_box`: the image bounding box
        """
        x_min = x_min - zoom_out
        if x_min < 0:
            x_min = 0
        y_min = y_min - zoom_out
        if y_min < 0:
            y_min = 0
        x_max = x_max + zoom_out
        if x_max > image.shape[1]:
            x_max = image.shape[1]
        if y_max > image.shape[0]:
            y_max = image.shape[0]
        y_max = y_max + zoom_out
        image_box = image[y_min:y_max,x_min:x_max,:]
        return image_box
    
    def run(self):
        """
        Run the shadow detection
        """
        for i in tqdm(range(len(self.target_files))):
            
            image_meta_path = self.target_files[i]
            image_path = image_meta_path[0]
            meta_path = image_meta_path[1]
            
            image = cv2.imread(image_path)
            
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            f.close()
            
            bboxes = None
            
            if self.groundtruth_only:
                if "groundtruth" not in meta:
                    continue
                else:
                    bboxes = meta["groundtruth"]["box"]
            elif "predicted" not in meta:
                continue
            elif "model_{}".format(self.model_number) not in meta["predicted"]:
                continue
            else:
                bboxes = meta["predicted"]["model_{}".format(self.model_number)]["box"]
            
            if not bboxes or len(bboxes) == 0:
                continue
            
            res_shadow = []
            
            for j in range(len(bboxes)):
                box = bboxes[j]
                
                x_min = min(box[0], box[2])
                y_min = min(box[1], box[3])
                x_max = min(box[0], box[2])
                y_max = min(box[1], box[3])
                
                image_box = self.box_zoom_out(image, x_min, y_min, x_max, y_max, self.zoom_out)
                
                shadow_detection = ShadowDetection(image_box,
                                                   minimum_size_window=self.minimum_size_window,
                                                   threshold_v=self.threshold_v,
                                                   threshold_s=self.threshold_s,
                                                   ratio=self.ratio,
                                                   d_0=self.d_0)
                res = shadow_detection.run(seed_only=True, verbose=0)
                res_shadow.append(res)
                
            # save the results inside the right key
            if self.groundtruth_only:
                meta["groundtruth"]["shadow"] = {}
                meta["groundtruth"]["shadow"]['zoom_out_{}'.format(self.zoom_out)] = res_shadow
            else:
                meta["predicted"]["model_{}".format(self.model_number)]["shadow"] = {}
                meta["predicted"]["model_{}".format(self.model_number)]["shadow"]['zoom_out_{}'.format(self.zoom_out)] = res_shadow
            
            with open(meta_path, 'w') as f:
                json.dump(meta, f, sort_keys=True, indent=4)
            f.close()
            

if __name__ == "__main__":
    image_folder = "C:\\Users\\AISG\\Documents\\Jonas\\Helipad\\Helipad_DataBase\\Helipad_DataBase_original"
    meta_folder = "C:\\Users\\AISG\\Documents\\Jonas\\Helipad\\Helipad_DataBase_meta\\Helipad_DataBase_meta_original"
    model_number = 7
    groundtruth_only = True
    tms = False 
    zoom_out = 5
    index_path = None
    minimum_size_window=3
    threshold_v=0.35
    threshold_s=0.02
    ratio=1
    d_0=3
    
    run_shadow_detection = RunShadowDetection(image_folder=image_folder, 
                                              meta_folder=meta_folder,
                                              model_number=model_number, 
                                              groundtruth_only=groundtruth_only, 
                                              tms=tms, 
                                              zoom_out=zoom_out, 
                                              index_path=index_path,
                                              minimum_size_window=minimum_size_window, 
                                              threshold_v=threshold_v, 
                                              threshold_s=threshold_s, 
                                              ratio=ratio, 
                                              d_0=d_0)
    
    run_shadow_detection.run()