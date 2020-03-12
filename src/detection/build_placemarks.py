import os
import json
from tqdm import tqdm as tqdm

from src.knn.knn_predict import KNNPredict


class BuildPlacemarks:

    def __init__(self, meta_folder, model_number, threshold, knn=True, index_path=None, model_name=None):

        self.meta_folder = meta_folder
        self.model_number = model_number
        self.threshold = threshold
        self.knn = knn
        self.model_name = model_name
        self.index_path = index_path

        self.output_name = "placemarks_m{}_t{}_{}.kml".format(model_number, threshold, model_name)

    def build_target_file(self):
        target = []
        if self.index_path:
            with open(self.index_path, 'r') as f:
                for line in f:
                    zoom, xtile, ytile, meta_filename = KNNPredict.get_meta_info_from_line(line)
                    meta_path = os.path.join(self.meta_folder, zoom, xtile, meta_filename)
                    target.append(meta_path)
        else:
            for subdirs, dirs, files in os.walk(self.meta_folder, topdown=True):
                for file in files:
                    filepath = os.path.join(subdirs, file)
                    target.append(filepath)
        return target

    def run(self):

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
                                if knn[k] == 0:
                                    continue
                            else:
                                continue
                        f.write('\t\t\t<Placemark>\n')
                        f.write('\t\t\t\t<name>Placemark {}</name>\n'.format(i))
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

    meta_folder = "C:\\Users\\AISG\\Documents\\Jonas\\Real_World_Dataset_TMS_meta\\sat\\"
    model_number = 7
    threshold = 0.999
    index_path = "../database_management/helipad_path_over_0.999.txt"

    build_placemarks = BuildPlacemarks(meta_folder,
                                       model_number,
                                       threshold,
                                       knn=True,
                                       model_name="cnn_validation",
                                       index_path=index_path)

    build_placemarks.run()