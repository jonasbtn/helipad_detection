import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm as tqdm


class MetaToCsvTMS:
    
    """
    Convert all the meta files inside the meta folder to one csv file
    """
    
    def __init__(self, meta_folder, output_folder, output_filename, model_number, filter_config, index_path=None):
        """
        `meta_folder`: string, path to the meta folder\n
        `output_folder`: string, path to the output folder. The folder must already exist.\n
        `output_filename`: string, filename of the output csv file\n
        `model_number`: int, number of the model\n
        `filter_config`: dict, filters configuration\n
        `index_path`: string, path to the file containing all the names of images with a bounding box\n
        """
        self.meta_folder = meta_folder
        self.output_folder = output_folder
        self.output_filename = output_filename
        self.model_number = model_number
        self.filter_config = filter_config
        self.index_path = index_path
        if self.index_path:
            self.target_files = self.convert_meta_filename_to_path(self.meta_folder, self.index_path)
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
                target_files.append(meta_path)
        return target_files
    
    @staticmethod
    def convert_meta_filename_to_path(meta_folder, index_path):
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
                image_meta_path.append(meta_path)
        f.close()
        return image_meta_path
    
    def run(self):
        """
        Run the conversion
        """
        j = 1
        
        columns = ['index', 'latitude', 'longitude', 'score', 'area', 'shadow'] 
        data = []
        
        for i in tqdm(range(len(self.target_files))):
            meta_filepath = self.target_files[i]
            with open(meta_filepath, 'r') as f:
                meta = json.load(f)
            
            
            prediction = meta['predicted']['model_{}'.format(self.model_number)]
            for k in range(len(prediction['coordinates']['center'])):
                index = j
                
                latitude = prediction['coordinates']['center'][k][0]
     
                longitude = prediction['coordinates']['center'][k][1]
                
                score = prediction['score'][k]
                if 'score' in self.filter_config:
                    if self.filter_config['score']['activate']:
                        if score<self.filter_config['score']['threshold']:
                            continue
                            
                area = prediction['coordinates']['area'][k]
                if 'area' in self.filter_config:
                    if self.filter_config['area']['activate']:
                        if area>self.filter_config['area']['higher'] or area<self.filter_config['area']['lower']:
                            continue
                    else:
                        area = None
                else:
                    area = None
                
                if 'shadow' in self.filter_config:
                    if self.filter_config['shadow']['activate']:
                        if 'zoom_out' in self.filter_config['shadow']:
                            zoom_out = self.filter_config['shadow']['zoom_out']
                            shadow = prediction['shadow']['zoom_out_{}'.format(zoom_out)][k]
                            if shadow:
                                continue
                        else:
                            shadow = None
                    else:
                        shadow = None
                else:
                    shadow = None
                   
                row = [index, latitude, longitude, score, area, shadow]
                data.append(row)
                j += 1
        
        df = pd.DataFrame(data=data, columns=columns)
        df.to_csv(os.path.join(self.output_folder, self.output_filename))
