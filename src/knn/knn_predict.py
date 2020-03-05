import os
import numpy as np
import cv2
import json
from tqdm import tqdm as tqdm
from sklearn.externals import joblib

from src.knn.knn_build_database import KNNBuildDatabase
from src.knn.knn_training import KNNTraining


class KNNPredict:

    def __init__(self, image_folder, meta_folder, model_number, knn_weights_filename,
                 mode="histogram", size=(64, 64), bins=(8, 8, 8)):

        self.image_folder = image_folder
        self.meta_folder = meta_folder
        self.model_number = model_number
        self.mode = mode
        self.knn = joblib.load(knn_weights_filename)

        self.knn_build_database = KNNBuildDatabase(self.image_folder, self.meta_folder, self.model_number, train=False)
        self.image_id = self.knn_build_database.image_id
        self.X = self.knn_build_database.X

        if mode == "raw_pixel":
            self.features = KNNTraining.dataset_to_matrix_features(self.X, size=size)
        elif mode == "histogram":
            self.features = KNNTraining.dataset_to_matrix_histogram(self.X, bins=bins)
        else:
            raise ValueError("mode is \'raw_pixel\' or \'histogram\'")

    def predict(self):
        self.y_predict = self.knn.predict(self.features)

    def write_prediction_to_meta(self):

        for i in tqdm(range(len(self.y_predict))):

            y_pred = self.y_predict[i]

            image_id = self.image_id[i]
            image_info = image_id.split("_")
            zoom = image_info[1]
            xtile = image_info[2]
            ytile = image_info[3]

            meta_path = os.path.join(self.meta_folder, zoom, image_id+".meta")

            with open(meta_path, 'r') as f:
                meta = json.load(f)

            if "predicted" not in meta:
                continue
            elif f'model_{self.model_number}' not in meta["predicted"]:
                continue
            else:
                meta["predicted"][f'model_{self.model_number}']["knn"] = y_pred

            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=4, sort_keys=True)









