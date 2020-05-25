import os
import json


class BenchmarkManagerTMS:
    
    def __init__(self, image_folder, meta_folder, model_number, index_path=None):
        self.image_folder = image_folder
        self.meta_folder = meta_folder
        self.model_number = model_number
        self.index_path = index_path
        self.tms = True
        print("Loading Files")
        if self.index_path:
            self.target_files = self.convert_meta_filename_to_path(self.image_folder, self.meta_folder, self.index_path)
        else:
            self.target_files = self.load_target_files()
        print("{} files loaded!".format(len(self.target_files)))

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
    
    def reinitialize_metrics(self):
        """
        Reinitialize the metrics to zeros.
        """
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
    
    def run(self, filters_config):
        
        self.filters_config = filters_config
        self.reinitialize_metrics()
        
        for i in range(len(self.target_files)):
            image_meta_path = self.target_files[i]
            image_path = image_meta_path[0]
            meta_path = image_meta_path[1]
            
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            f.close()
            
            model_pred = meta['predicted'][f'model_{self.model_number}']
            
            res_after_filters = []
            
            for k in range(len(model_pred['box'])):
                
                res = True
                
                # filter first by shadows
                if 'shadow' in self.filters_config:
                        if 'activate' in self.filters_config['shadow']:
                            if self.filters_config['shadow']['activate']:
                                if 'zoom_out' in self.filters_config['shadow']:
                                    zoom_out = self.filters_config['shadow']['zoom_out']
                                else:
                                    zoom_out = 0
                                shadow = model_pred['shadow']['zoom_out_{}'.format(zoom_out)][k]
                                if shadow:
                                    res = False
                
                # filter then by areas
                if 'area' in self.filters_config:
                        if 'activate' in self.filters_config['area']:
                            if self.filters_config['area']['activate']:
                                if 'lower' in self.filters_config['area']:
                                    lower = self.filters_config['area']['lower']
                                else:
                                    lower = 0
                                if 'higher' in self.filters_config['area']:
                                    higher = self.filters_config['area']['higher']
                                else:
                                    higher = 2000
                                
                                area = model_pred['coordinates']['area'][k]
                                if area < lower or area > higher:
                                    res = False
                
                # filter by score
                if 'score' in self.filters_config:
                    if 'activate' in self.filters_config['score']:
                        if self.filters_config['score']['activate']:
                            score_threshold = self.filters_config['score']['threshold']
                            score = model_pred["score"][k]
                            if score < score_threshold:
                                res = False
                
                res_after_filters.append(res)
        
            for k in range(len(res_after_filters)):
                res = res_after_filters[k]
                groundtruth = model_pred['groundtruth'][k]
                
                if res and groundtruth:
                    self.TP += 1
                elif res and not groundtruth:
                    self.FP += 1
                elif not res and groundtruth:
                    self.FN += 1
                elif not res and not groundtruth:
                    self.TN += 1
        
        res = {'TP': self.TP, 
               'TN': self.TN,
               'FP': self.FP,
               'FN': self.FN}

        accuracy = (res['TP']+res['TN'])/(res['TP']+res['TN']+res['FP']+res['FN'])
        error = (res['FP']+res['FN'])/(res['TP']+res['TN']+res['FP']+res['FN'])
        precision = res['TP']/(res['TP']+res['FP'])
        recall = res['FP']/(res['TN']+res['FP'])
        res['accuracy'] = accuracy
        res['error'] = error
        res['precision'] = precision
        res['recall'] = recall
        
        self.filters_config['result_benchmark'] = res
        
        return self.filters_config