import os
import numpy as np
import cv2
import json
from tqdm import tqdm as tqdm
from sklearn.externals import joblib

from knn_build_database import KNNBuildDatabase
from knn_training import KNNTraining


class KNNPredict:

    def __init__(self, image_folder, meta_folder, index_path_filename, model_number, knn_weights_filename,
                 mode="histogram", size=(64, 64), bins=(8, 8, 8)):

        self.image_folder = image_folder
        self.meta_folder = meta_folder
        self.index_path_filename = index_path_filename
        self.model_number = model_number
        self.mode = mode
        self.knn = joblib.load(knn_weights_filename)

        self.X = []
        self.image_id = []

        # self.knn_build_database = KNNBuildDatabase(self.image_folder, self.meta_folder, self.model_number, train=False, TMS=True)
        # self.knn_build_database.run()
        # self.image_id = self.knn_build_database.image_id
        # self.X = self.knn_build_database.X
        self.load_image_to_predict()

        print(len(self.X))

        if mode == "raw_pixel":
            self.features = KNNTraining.dataset_to_matrix_features(self.X, size=size)
        elif mode == "histogram":
            self.features = KNNTraining.dataset_to_matrix_histogram(self.X, bins=bins)
        else:
            raise ValueError("mode is \'raw_pixel\' or \'histogram\'")

        print(self.features.shape)

    def load_image_to_predict(self):
        with open(self.index_path_filename, 'r') as f:
            for line in f:
                meta_filename = line
                meta_info = os.path.splitext(meta_filename)[0].split('_')
                zoom = meta_info[1]
                xtile = meta_info[2]
                ytile = meta_info[3]
                image_path = os.path.join(self.image_folder, zoom, xtile, str(ytile)+".jpg")
                meta_path = os.path.join(self.meta_folder, zoom, meta_filename)
                image = cv2.imread(image_path)
                with open(meta_path, 'r') as f_meta:
                    meta = json.load(f_meta)
                predicted = meta["predicted"][f'model_{self.model_number}']
                bboxes_predicted = predicted["box"]
                for box_predicted in bboxes_predicted:
                    x_min = box_predicted[0]
                    y_min = box_predicted[1]
                    x_max = box_predicted[2]
                    y_max = box_predicted[3]
                    image_box = image[x_min:x_max, y_min:y_max, :]
                    self.X.append(image_box)
                    self.image_id.append(os.path.splitext(meta_filename)[0])

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


if __name__ == "__main__":

    image_folder = "C:\\Users\\AISG\\Documents\\Jonas\\Real_World_Dataset_TMS\\sat"
    meta_folder = "C:\\Users\\AISG\\Documents\\Jonas\\Real_World_Dataset_TMS_meta\\sat"
    index_path_score_filename = "C:\\Users\\AISG\\Documents\\Jonas\\helipad_detection\\src\\database_management\\helipad_path_over_0.999.txt"
    model_number = 7
    knn_weights_filename = "knn_histogram_2.pkl"

    knn_predict = KNNPredict(image_folder, meta_folder, model_number, knn_weights_filename)

    knn_predict.predict()


# ip : 172.17.136.194






