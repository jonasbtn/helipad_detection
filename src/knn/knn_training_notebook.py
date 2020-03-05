##

from src.knn.knn_build_database import KNNBuildDatabase

image_folder = "C:\\Users\\jonas\\Desktop\\Helipad\\Helipad_DataBase\\Helipad_DataBase_original"
meta_folder = "C:\\Users\\jonas\\Desktop\\Helipad\\Helipad_DataBase_meta\\Helipad_DataBase_meta_original"
model_number = 7

knn_build_database = KNNBuildDatabase(image_folder, meta_folder, model_number)

knn_build_database.run()

##

from collections import Counter
counter = Counter(knn_build_database.y)

counter

##

X = knn_build_database.X
y = knn_build_database.y

##

shapes_x = []
shapes_y = []

for x in X:
    shapes_x.append(x.shape[0])
    shapes_y.append(x.shape[1])

print(min(shapes_x))
print(min(shapes_y))

# --> 64 x 64
##
import cv2
import numpy as np
from tqdm import tqdm as tqdm
##

def image_to_feature_vector(image, size=(64, 64)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()

def dataset_to_matrix_features(X, size=(64,64)):
    matrix = []
    for i in tqdm(range(len(X))):
        x = X[i]
        x_flat = image_to_feature_vector(x, size=size)
        matrix.append(x_flat)
    return np.array(matrix)

##

matrix = dataset_to_matrix_features(X, size=(64,64))

##

matrix.shape

##

from sklearn.model_selection import train_test_split

##

# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
(trainRI, testRI, trainRL, testRL) = train_test_split(
	matrix, y, test_size=0.25, random_state=42)

##

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=13, n_jobs=-1)

knn.fit(trainRI, trainRL)

##

acc = knn.score(testRI, testRL)
print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))

##

import imutils

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

def dataset_to_matrix_histogram(X):
    matrix = []
    for i in tqdm(range(len(X))):
        x = X[i]
        x_flat = extract_color_histogram(x)
        matrix.append(x_flat)
    return np.array(matrix)

##

histograms = dataset_to_matrix_histogram(X)

##

histograms.shape

##

(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
	histograms, y, test_size=0.25, random_state=42)

##

# train and evaluate a k-NN classifer on the histogram
# representations
print("[INFO] evaluating histogram accuracy...")
model = KNeighborsClassifier(n_neighbors=13,
	n_jobs=-1)
model.fit(trainFeat, trainLabels)

##

acc = model.score(testFeat, testLabels)
print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))


##

# --> don't distinguish between categories but just between helipad or false positive

y_binary = []
for target in y:
    if target == 12:
        y_binary.append(0)
    else:
        y_binary.append(1)

##

# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
(trainRI, testRI, trainRL, testRL) = train_test_split(
	matrix, y_binary, test_size=0.25, random_state=42)

##

from sklearn.neighbors import KNeighborsClassifier

knn_binairy = KNeighborsClassifier(n_neighbors=2, n_jobs=-1)

knn_binairy.fit(trainRI, trainRL)

##

acc = knn_binairy.score(testRI, testRL)
print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))

##

(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
	histograms, y_binary, test_size=0.25, random_state=42)

##

# train and evaluate a k-NN classifer on the histogram
# representations
print("[INFO] evaluating histogram accuracy...")
model_binairy = KNeighborsClassifier(n_neighbors=13,
	n_jobs=-1)
model_binairy.fit(trainFeat, trainLabels)

##

acc = model_binairy.score(testFeat, testLabels)
print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))

# --> 75.83%

##

from sklearn.externals import joblib

_ = joblib.dump(model_binairy, "src/knn/knn_binairy_histogram.pkl", compress=9)

##

from src.knn.knn_training import KNNTraining

##

knn_training = KNNTraining(nb_neighbors=2, nb_jobs=-1, test_size=0.25)

knn_training.fit(knn_build_database.X, knn_build_database.y, mode="histogram", binary=True)

knn_training.score()

knn_training.save()
