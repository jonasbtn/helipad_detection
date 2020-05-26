import pandas as pd
import numpy as np
import os
import json
from urllib import parse, request


class CsvToMeta:
    
    """
    Initiate the meta folder from the CSV file containing the dataset.
    """
    
    def __init__(self, input_csv_file_path, output_folder, download_images=False, google_api_key_filepath=None):
        """
        `input_csv_file_path`: string, path to the csv file containing the dataset informations\n
        `output_folder`: string, path to the folder where to store the dataset. 2 subfolders are created: `Helipad_Dataset_meta` and `Helipad_Dataset`\n
        `download_images`: boolean, True to download and store the images from Google Maps\n
        `google_api_key_filepath`: string, path to the local file containing the user's Google Maps Api key.
        """
        self.input_csv_file_path = input_csv_file_path
        self.output_folder = output_folder
        self.download_images = download_images
        
        if not os.path.isdir(os.path.join(self.output_folder, 'Helipad_Dataset_meta')):
            os.mkdir(os.path.join(self.output_folder, 'Helipad_Dataset_meta'))
        if not os.path.isdir(os.path.join(self.output_folder, 'Helipad_Dataset_meta', 'Helipad_Dataset_meta_original')):
            os.mkdir(os.path.join(self.output_folder, 'Helipad_Dataset_meta', 'Helipad_Dataset_meta_original'))
        if download_images:
            if not os.path.isdir(os.path.join(self.output_folder, 'Helipad_Dataset')):
                os.mkdir(os.path.join(self.output_folder, 'Helipad_Dataset'))
            if not os.path.isdir(os.path.join(self.output_folder, 'Helipad_Dataset', 'Helipad_Dataset_original')):
                os.mkdir(os.path.join(self.output_folder, 'Helipad_Dataset', 'Helipad_Dataset_original'))
            with open(google_api_key_filepath, 'r') as f:
                self.api_key = f.read()
            f.close()
        
        self.image_folder = os.path.join(self.output_folder, 'Helipad_Dataset', 'Helipad_Dataset_original')
        self.meta_folder = os.path.join(self.output_folder, 'Helipad_Dataset_meta', 'Helipad_Dataset_meta_original')
        
    def run(self):
        
        df = pd.read_csv(self.input_csv_file_path)
        
        helipad_numbers = df.Helipad_number.value_counts().keys().sort_values().tolist()
        
        for helipad_number in helipad_numbers:
            
            index = np.where(df['Helipad_number'].values == helipad_number)[0]
            df_helipad = df.iloc[index, :]
            
            nb_helipad = df_helipad.shape[0]
            
            if nb_helipad > 0:
                meta = dict()
                meta['coordinates'] = {}
                meta['coordinates']['latitude'] = float(df_helipad.iloc[0, :]['latitude'])
                meta['coordinates']['longitude'] = float(df_helipad.iloc[0, :]['longitude'])
                meta['coordinates']['zoom'] = int(df_helipad.iloc[0, :]['zoom'])
                meta['info'] = {}
                meta['info']['FAA_index'] = int(df_helipad.iloc[0, :]['FAA_index'])
                meta['info']['Helipad_number'] = int(df_helipad.iloc[0, :]['Helipad_number'])
                meta['info']['url'] = str(df_helipad.iloc[0, :]['url'])
                meta['groundtruth'] = {}
                meta['groundtruth']['helipad'] = bool(df_helipad.iloc[0, :]['groundtruth'])
                meta['groundtruth']['category'] = str(df_helipad.iloc[0, :]['category'])
                bboxes = []
                for i in range(nb_helipad):
                    box = [float(df_helipad.iloc[i, :]['minX']),\
                           float(df_helipad.iloc[i, :]['minY']),\
                           float(df_helipad.iloc[i, :]['maxX']),\
                           float(df_helipad.iloc[i, :]['maxY'])]
                    bboxes.append(box)
                meta['groundtruth']['box'] = bboxes
                
                folder_id = helipad_number // 100
                
                folder_name = 'Folder_{:03d}'.format(folder_id)
                meta_filename = "Helipad_{:05d}.meta".format(helipad_number)
                meta_path = os.path.join(self.meta_folder, folder_name, meta_filename)
                
                if not os.path.isdir(os.path.dirname(meta_path)):
                    os.mkdir(os.path.dirname(meta_path))
                
                with open(meta_path, 'w') as f:
                    json.dump(meta, f, indent=4, sort_keys=True)
                
                print('Saved :' + meta_path)
                
                if self.download_images:
                    image_filename = "Helipad_{:05d}.png".format(helipad_number)
                    image_path = os.path.join(self.image_folder, folder_name, image_filename)
                    if not os.path.isdir(os.path.dirname(image_path)):
                        os.mkdir(os.path.dirname(image_path))
                    url = df_helipad.iloc[0, :]['url']
                    url = url.replace('APIKEY', self.api_key)
                    
                    request.urlretrieve(url, image_path)
                    print('Downloaded: ' + image_path)
        
        print('Dataset created!')
    

if __name__ == "__main__":
    input_csv_file_path = "C:\\Users\\AISG\\Documents\\Jonas\\Helipad\\Helipad_DataBase_meta\\Helipad_DataBase_annotated.csv"
    output_folder = "C:\\Users\\AISG\\Documents\\Jonas\\New_Dataset_Folder"
    download_images = False
    google_api_key_filepath = None

    csv_to_meta = CsvToMeta(input_csv_file_path=input_csv_file_path,
                           output_folder=output_folder,
                           download_images=download_images,
                           google_api_key_filepath=google_api_key_filepath)

    csv_to_meta.run()