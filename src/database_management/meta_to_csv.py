import pandas as pd
import numpy as np
import os
import json


class MetaToCsv:
    
    """
    Convert all the meta files inside the meta folder to one csv file
    """
    
    def __init__(self, meta_folder, output_folder, output_filename):
        """
        `meta_folder`: string, path to the meta folder\n
        `output_folder`: string, path to the output folder. The folder must already exist.\n
        `output_filename`: string, filename of the output csv file\n
        """
        self.meta_folder = meta_folder
        self.output_folder = output_folder
        self.output_filename = output_filename
    
    def run(self):
        """
        Run the conversion
        """
        data = []
        for subdir, dirs, files in os.walk(self.meta_folder, topdown=True):
            for file in files:
                if file[0] == ".":
                    continue
                if os.path.splitext(file)[1] != ".meta":
                    continue
                filepath = os.path.join(subdir, file)
                with open(filepath, 'r') as f:
                    meta = json.load(f)
                f.close()
                info = meta['info']
                coordinates = meta['coordinates']
                if 'groundtruth' not in meta:
                    continue
                groundtruth = meta['groundtruth']    
                if not groundtruth['helipad']:
                    helipad = [info['Helipad_number'], info['FAA_index'], coordinates['latitude'], coordinates['longitude'],\
                               groundtruth['helipad'], str(-1), -1, -1, -1, -1, info['url']]
                    data.append(helipad)
                else:
                    if 'box' not in groundtruth or len(groundtruth['box']) == 0:
                        if 'category' not in groundtruth:
                            helipad = [info['Helipad_number'], info['FAA_index'], coordinates['latitude'], coordinates['longitude'],\
                               groundtruth['helipad'], str(-1), -1, -1, -1, -1, info['url']]
                        else:
                            cat = groundtruth['category']
                            if cat == 'd':
                                cat = "other"
                            elif cat == 'u':
                                cat = "unknown"
                            helipad = [info['Helipad_number'], info['FAA_index'], coordinates['latitude'], coordinates['longitude'],\
                               groundtruth['helipad'], cat, -1, -1, -1, -1, info['url']]
                        data.append(helipad)
                    else:
                        cat = groundtruth['category']
                        if cat == 'd':
                            cat = "other"
                        elif cat == 'u':
                            cat = "unknown"
                        for i in range(len(groundtruth['box'])):
                            box = groundtruth['box'][i]
                            helipad = [info['Helipad_number'], info['FAA_index'], coordinates['latitude'], coordinates['longitude'],\
                               groundtruth['helipad'], cat, \
                                       min(int(box[0]), int(box[2])),\
                                       min(int(box[1]), int(box[3])),\
                                       max(int(box[0]), int(box[2])),\
                                       max(int(box[1]), int(box[3])), info['url']]
                            data.append(helipad)

        columns = ['Helipad_number', 'FAA_index', 'latitude', 'longitudes', 'groundtruth', 'category', 'minX', 'minY', 'maxX', 'maxY', 'url']

        df = pd.DataFrame(data=data, columns=columns)

        df.to_csv(os.path.join(self.output_folder, self.output_filename))

        print('Conversion done and file saved!')


if __name__ == "__main__":
    
    meta_folder_original = "C:\\Users\\AISG\\Documents\\Jonas\\Helipad\\Helipad_DataBase_meta\\Helipad_DataBase_meta_original"
    output_folder = "C:\\Users\\AISG\\Documents\\Jonas\\Helipad\\Helipad_DataBase_meta"
    output_filename = "Helipad_DataBase_annotated.csv"
    
    meta_to_csv = MetaToCsv(meta_folder=meta_folder_original,
                            output_folder=output_folder,
                            output_filename=output_filename)
    
    meta_to_csv.run()