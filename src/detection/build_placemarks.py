import os
import json
from tqdm import tqdm as tqdm
import pathlib


class BuildPlacemarks:
    
    """
    Build a placemarks file containing the coordinates of all the bounding boxe detected. This allows to visualize faster all the detection over an entire area inside the software SAS Planet. The placemarks are saved into the folder "placemarks" inside the same folder as this script. 
    """

    def __init__(self, meta_folder, model_number, threshold, knn=True, index_path=None, model_name=None,
                 model_validation_threshold=None, prefix=""):
        """
        `meta_folder`: string, path to the folder containing the meta files\n
        `model_number`: int, number of the model. The detection has to be run on the images by the model first.\n
        `threshold`: float, score threshold, all the detected bounding boxes below this threshold won't be included. \n
        `knn`: boolean, True to consider the second model validation of the bounding boxes\n
        `index_path`: string, path of to an index created by `database_management.index_path_by_score` if existed, else None\n
        `model_name`: string, validation model name (the according key inside the meta file)\n
        `model_validation_threshold`: float, score threshold of the validating model\n
        `prefix`: string, output file prefix (ie: `"Manilla_"`)\n
        """
        self.meta_folder = meta_folder
        self.model_number = model_number
        self.threshold = threshold
        self.knn = knn
        self.model_name = model_name
        self.index_path = index_path
        self.model_validation_threshold = model_validation_threshold
        self.output_name = os.path.join(pathlib.Path(__file__).parent.absolute(), "placemarks", "{}placemarks_m{}_t{}_v{}_{}_t{}.kml".format(prefix, model_number, threshold, knn, model_name, model_validation_threshold))
    
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
                scores = meta["predicted"][key]["score"]

                for k in range(len(lat_long_centers)):
                    lat_long = lat_long_centers[k]
                    if scores[k] > self.threshold:
                        if self.knn:
                            if self.model_name in meta["predicted"][key]:
                                knn = meta["predicted"][key][self.model_name]
                                if self.model_validation_threshold:
                                    if knn[k] < self.model_validation_threshold:
                                        continue
                                elif knn[k] == 0:
                                    continue
                            else:
                                continue
                        f.write('\t\t\t<Placemark>\n')
                        if self.knn:
                            f.write('\t\t\t\t<name>Placemark {}-{}-{}</name>\n'.format(i, str(scores[k])[:5], str(knn[k])[2:5]))
                        else:
                            f.write('\t\t\t\t<name>Placemark {}-{}</name>\n'.format(i, str(scores[k])[2:5]))
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
    threshold = 0
    index_path = "../database_management/helipad_path_over_0.txt"

    build_placemarks = BuildPlacemarks(meta_folder,
                                       model_number,
                                       threshold,
                                       knn=True,
                                       model_name="cnn_validation",
                                       model_validation_threshold=0,
                                       index_path=index_path)

    build_placemarks.run()