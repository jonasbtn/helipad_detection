import os
import json

# TODO : Add a run example in the main below


class BBComputeAreaTMS:
    
    """
    Compute the area of the bounding box in m^3 and save inside the meta file in key :\n
    `meta["predicted"]["model_{}".format(model_number)]["coordinates"]["area"]`\n
    The dataset has to be under the TMS' stucture\n
    """
    
    def __init__(self, image_folder, meta_folder, model_number, index_path=None):
        """
        `image_folder`: string, path to the image folder\n
        `meta_folder`: string, path to the meta folder\n
        `model_number`: int, number of the model that predicted to bounding boxes\n
        `index_path`: string or None, path to the index files containing the names of the files with a bounding box inside\n
        """
        self.image_folder = image_folder
        self.meta_folder = meta_folder
        self.model_number = model_number
        self.index_path = index_path
        self.tms = True
        if self.index_path:
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
    def compute_distance_between_two_gps_points(point_1, point_2):
        """
        Compute the distance in meters between two GPS coordinates\n
        `point_1`: tuple (latitude, longitude)\n
        `point_2`: tuple (latitude, longitude)\n
        Returns\n
        `distance`: float, distance between the points in meters
        """
        # approximate radius of earth in km
        R = 6373.0

        lat1 = radians(point_1[0])
        lon1 = radians(point_1[1])
        lat2 = radians(point_2[0])
        lon2 = radians(point_2[1])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance = R * c * 1000

        return distance
    
    def run(self):
        """
        Run the Area computation
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
            
            if "predicted" not in meta:
                continue
            elif "model_{}".format(self.model_number) not in meta["predicted"]:
                continue
            elif len(meta["predicted"]["model_{}".format(self.model_number)]["box"]) == 0:
                continue
            
            bounds = meta["predicted"]["model_{}".format(model_number)]["coordinates"]["bounds"]
            
            area = []
            
            for j in range(len(bounds)):
                bound = bounds[j]
                l = self.compute_distance_between_two_gps_points(bound[0], bound[1])
                L = self.compute_distance_between_two_gps_points(bound[0], bound[2])
                area = l*L
                areas.append(area)
                
            # save the results
            meta["predicted"]["model_{}".format(model_number)]["coordinates"]["area"] = areas
            
            with open(meta_path, 'w') as f:
                json.dump(meta, f, sort_keys=True, indent=4)
            f.close()

    
if __name__ == "__main__:
    
            
    