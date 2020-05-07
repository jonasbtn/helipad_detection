import cv2
import imutils
import numpy as np
from tqdm import tqdm as tqdm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib


class KNNTraining:

    def __init__(self, nb_neighbors=2, nb_jobs=-1, test_size=0.25):
        self.nb_neighbors = nb_neighbors
        self.nb_jobs = nb_jobs
        self.test_size = test_size

        self.model = KNeighborsClassifier(n_neighbors=nb_neighbors, n_jobs=nb_jobs)
        # self.model = RandomForestClassifier(n_estimators=100)

    @staticmethod
    def convert_label_to_binary(y):
        y_binary = []
        for target in y:
            if target == 12:
                y_binary.append(0)
            else:
                y_binary.append(1)
        return y_binary

    @staticmethod
    def image_to_feature_vector(image, size=(64, 64)):
        # resize the image to a fixed size, then flatten the image into
        # a list of raw pixel intensities
        return cv2.resize(image, size).flatten()

    @staticmethod
    def dataset_to_matrix_features(X, size=(64, 64)):
        matrix = []
        for i in tqdm(range(len(X))):
            x = X[i]
            x_flat = KNNTraining.image_to_feature_vector(x, size=size)
            matrix.append(x_flat)
        return np.array(matrix)

    @staticmethod
    def extract_color_histogram(image, bins=(8, 8, 8)):
        # extract a 3D color histogram from the HSV color space using
        # the supplied number of `bins` per channel
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                            [0, 180, 0, 256, 0, 256])
        # handle normalizing the histogram if we are using OpenCV 2.4.X
        if imutils.is_cv2():
            hist = cv2.normalize(hist)
        # otherwise, perform "in place" normalization in OpenCV 3 (I
        # personally hate the way this is done
        else:
            cv2.normalize(hist, hist)
        # return the flattened histogram as the feature vector
        return hist.flatten()

    @staticmethod
    def dataset_to_matrix_histogram(X, bins=(8, 8, 8)):
        matrix = []
        for i in tqdm(range(len(X))):
            x = X[i]
            x_flat = KNNTraining.extract_color_histogram(x, bins=bins)
            matrix.append(x_flat)
        return np.array(matrix)

    @staticmethod
    def min_shape(X):
        shapes_x = []
        shapes_y = []
        for x in X:
            shapes_x.append(x.shape[0])
            shapes_y.append(x.shape[1])
        print(min(shapes_x))
        print(min(shapes_y))

    def fit(self, X, y, mode, binary=True, size=(64, 64), bins=(8, 8, 8)):
        if mode == "raw_pixel":
            features = self.dataset_to_matrix_features(X, size=size)
        elif mode == "histogram":
            features = self.dataset_to_matrix_histogram(X, bins=bins)
        else:
            raise ValueError("mode is \'raw_pixel\' or \'histogram\'")

        self.mode = mode

        if binary:
            y = KNNTraining.convert_label_to_binary(y)

        (self.trainFeat, self.testFeat, self.trainLabels, self.testLabels) = train_test_split(
            features, y, test_size=self.test_size, random_state=42)

        self.model.fit(self.trainFeat, self.trainLabels)

    def score(self):
        acc = self.model.score(self.testFeat, self.testLabels)
        print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))

    def save(self, model_number, dataset):
        _ = joblib.dump(self.model, "knn_{}_{}_{}_{}.pkl".format(self.mode, self.nb_neighbors, model_number, dataset), compress=9)
        # _ = joblib.dump(self.model, "random_forest_e100.pkl".format(self.mode, self.nb_neighbors), compress=9)






