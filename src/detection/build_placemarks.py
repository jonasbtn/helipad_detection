import os
import json
from tqdm import tqdm as tqdm
import pathlib


class BuildPlacemarks:
    
    """
    Build a placemarks file containing the coordinates of all the bounding boxe detected. This allows to visualize faster all the detection over an entire area inside the software SAS Planet. The placemarks are saved into the folder "placemarks" inside the same folder as this script. 
    """

    def __init__(self, meta_folder, model_number, index_path=None, filters_config=None, prefix=""):
        """
        `meta_folder`: string, path to the folder containing the meta files\n
        `model_number`: int, number of the model. The detection has to be run on the images by the model first.\n
        `model_name`: string, validation model name (the according key inside the meta file)\n
        `filters_config`: json dict, filters configuration to activate. See example inside the main below\n
        `prefix`: string, output file prefix (ie: `"Manilla_"`)\n
        """
        self.meta_folder = meta_folder
        self.model_number = model_number
        self.index_path = index_path
        self.filters_config = filters_config
        self.prefix = prefix
        self.output_name = self.get_output_filename()
    
    def get_output_filename(self):
        output_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "placemarks")
        output_filename = "{}placemarks_m{}".format(self.prefix, self.model_number)
        if 'score' in self.filters_config:
            if 'activate' in self.filters_config['score']:
                if self.filters_config['score']['activate']:
                    score_threshold = self.filters_config['score']['threshold']
                    output_filename += "t_{}".format(score_threshold)
        if 'shadow' in self.filters_config:
            if 'activate' in self.filters_config['shadow']:
                if self.filters_config['shadow']['activate']:
                    if 'zoom_out' in self.filters_config['shadow']:
                        zoom_out = self.filters_config['shadow']['zoom_out']
                    else:
                        zoom_out = 0
                    output_filename += "s_zo{}".format(zoom_out)
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
                    output_filename += "a_l{}_h{}".format(lower, higher)
        if 'cnn_validation' in self.filters_config:
            if 'activate' in self.filters_config['cnn_validation']:
                if self.filters_config['cnn_validation']['activate']:
                    score_threshold = self.filters_config['cnn_validation']['threshold']
                    output_filename += "cnnv_t{}".format(score_threshold)
        output_filename += ".kml"
        output_filepath = os.path.join(output_dir, output_filename)
        return output_filepath
        
    @staticmethod
    def get_meta_info_from_line(line):
        """
        Get the image location info from the meta filename\n
        `line`: meta file name\n
        Returns\n
        `zoom`: Zoom level in the TMS coordinates\n
        `xtile`: XTile in the TMS coordinates\n
        `ytile`: YTile in the TMS coordinates\n
        `meta_filename`: the meta filename\n
        """
        if "\n" in line:
            meta_filename = line[:len(line) - 1]
        else:
            meta_filename = line
        meta_info = os.path.splitext(meta_filename)[0].split('_')
        zoom = meta_info[1]
        xtile = meta_info[2]
        ytile = meta_info[3]
        return zoom, xtile, ytile, meta_filename
    
    def build_target_file(self):
        """
        Build a list of meta file paths targeted for placemarks.
        """
        target = []
        if self.index_path:
            with open(self.index_path, 'r') as f:
                for line in f:
                    zoom, xtile, ytile, meta_filename = self.get_meta_info_from_line(line)
                    meta_path = os.path.join(self.meta_folder, zoom, xtile, meta_filename)
                    target.append(meta_path)
        else:
            for subdirs, dirs, files in os.walk(self.meta_folder, topdown=True):
                for file in files:
                    filepath = os.path.join(subdirs, file)
                    target.append(filepath)
        return target

    def run(self):
        """
        Run the creation of the placemarks
        """
        target_path = self.build_target_file()

        with open(self.output_name, 'w') as f:

            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<kml xmlns="http://earth.google.com/kml/2.2">\n')
            f.write('\t<Document>\n')
            f.write('\t\t<Folder>\n\t\t\t<name>New Category</name>\n')
            f.write('\t\t\t<open>1</open>\n')
            f.write('\t\t\t<Style>\n')
            f.write('\t\t\t\t<ListStyle>\n')
            f.write('\t\t\t\t\t<listItemType>check</listItemType>\n')
            f.write('\t\t\t\t\t<bgColor>00ffffff</bgColor>\n')
            f.write('\t\t\t\t</ListStyle>\n')
            f.write('\t\t\t</Style>\n')

            i = 0

            for i in tqdm(range(len(target_path))):
                
                filepath = target_path[i]
                with open(filepath, 'r') as j:
                    meta = json.load(j)
                if "predicted" not in meta:
                    continue
                key = "model_{}".format(self.model_number)
                if key not in meta["predicted"]:
                    continue
                lat_long_centers = meta["predicted"][key]["coordinates"]["center"]
                
                for k in range(len(lat_long_centers)):
                    
                    placemark_name = "Placemark_{}".format(i)
                    
                    # filter first by shadows
                    if 'shadow' in self.filters_config:
                        if 'activate' in self.filters_config['shadow']:
                            if self.filters_config['shadow']['activate']:
                                if 'zoom_out' in self.filters_config['shadow']:
                                    zoom_out = self.filters_config['shadow']['zoom_out']
                                else:
                                    zoom_out = 0
                                shadow = meta["predicted"][key]['shadow']['zoom_out_{}'.format(zoom_out)][k]
                                if shadow:
                                    continue
                                else:
                                    placemark_name += "_s{}".format(shadow)
                    
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
                                
                                area = meta["predicted"][key]['coordinates']['area'][k]
                                if area > lower and area < higher:
                                    placemark_name += "_a{}".format(area)
                                else:
                                    continue
                    
                    # filter by score
                    if 'score' in self.filters_config:
                        if 'activate' in self.filters_config['score']:
                            if self.filters_config['score']['activate']:
                                score_threshold = self.filters_config['score']['threshold']
                                score =  meta["predicted"][key]["score"][k]
                                if score < score_threshold:
                                    continue
                                else:
                                    placemark_name += "_{}".format(str(score)[2:5])
                    
                    # filter by cnn validation score
                    if 'cnn_validation' in self.filters_config:
                        if 'activate' in self.filters_config['cnn_validation']:
                            if self.filters_config['cnn_validation']['activate']:
                                score_threshold = self.filters_config['cnn_validation']['threshold']
                                if "cnn_validation" in meta["predicted"]["key"]:
                                    score = meta["predicted"]["key"]["cnn_validation"][k]
                                    if  score < score_threshold:
                                        continue
                                    else:
                                        placemark_name += "_{}".format(str(score)[2:5])
                   
                    f.write('\t\t\t<Placemark>\n')
                    f.write('\t\t\t\t<name>{}</name>\n'.format(placemark_name))
                    f.write('\t\t\t\t<description>1/7/2020 2:18:10 PM</description>\n')
                    f.write('\t\t\t\t<Style>\n')
                    f.write('\t\t\t\t\t<LabelStyle>\n')
                    f.write('\t\t\t\t\t\t<color>A600FFFF</color>\n')
                    f.write('\t\t\t\t\t\t<scale>1</scale>\n')
                    f.write('\t\t\t\t\t</LabelStyle>\n')
                    f.write('\t\t\t\t\t<IconStyle>\n')
                    f.write('\t\t\t\t\t\t<scale>0.5</scale>\n')
                    f.write('\t\t\t\t\t\t<Icon>\n')
                    f.write('\t\t\t\t\t\t\t<href>files/1.png</href>\n')
                    f.write('\t\t\t\t\t\t</Icon>\n')
                    f.write('\t\t\t\t\t\t<hotSpot x="0.5" y="0" xunits="fraction" yunits="fraction"/>\n')
                    f.write('\t\t\t\t\t</IconStyle>\n')
                    f.write('\t\t\t\t</Style>\n')
                    f.write('\t\t\t\t<Point>\n')
                    f.write('\t\t\t\t\t<extrude>1</extrude>\n')
                    f.write('\t\t\t\t\t<coordinates>{},{},0 </coordinates>\n'.format(lat_long[1], lat_long[0]))
                    f.write('\t\t\t\t</Point>\n')
                    f.write('\t\t\t</Placemark>\n')

                    i += 1

            f.write('\t\t</Folder>\n')
            f.write('\t</Document>\n')
            f.write('</kml>\n')

            f.close()


if __name__ == "__main__":

    meta_folder = "C:\\Users\\AISG\\Documents\\Jonas\\Real_World_Dataset_TMS_meta_save_2\\Real_World_Dataset_TMS_meta\\sat\\"
    model_number = 7
    index_path = "../database_management/helipad_path_over_0.txt"
    
    filters_config = { 
                  'shadow': {
                      'activate': True,
                      'zoom_out': 5
                    },
                   'area' : {
                       'activate': True,
                       'lower': 200,
                       'higher': 600,
                   },
                    'score': {
                        'activate': True,
                        'threshold': 0.99
                    },
                    'cnn_validation': {
                        'activate': False,
                        'threshold': 0
                    }
                }
                            
    prefix = "Manilla_"

    build_placemarks = BuildPlacemarks(meta_folder=meta_folder,
                                       model_number=model_number,
                                       index_path=index_path,
                                       filters_config=filters_config,
                                       prefix=prefix)

    build_placemarks.run()