import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import urllib.parse
import urllib.request
import re
import time
import os
import json
import shutil
import cv2

from shutil import copyfile


class CenterHelipads:
    
    """
    Center the helipad inside the image to get more precise GPS coordinates.
    """

    def __init__(self, image_folder, meta_folder):

        self.image_folder = image_folder
        self.meta_folder = meta_folder

    # get the center of the box from meta file
    def get_center_box(self, meta):
        if "groundtruth" in meta:
            if "box" in meta["groundtruth"]:
                boxes = meta["groundtruth"]["box"]

                centers = []

                for box in boxes:

                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                    x_c = (x1 + x2) // 2
                    y_c = (y1 + y2) // 2

                    centers.append((x_c, y_c))

                return centers
            else:
                return []
        else:
            return []

    def get_proportion_centered(self):
        nb_helipads = 0
        nb_helipads_centered = 0

        for subdir, dirs, files in os.walk(self.meta_folder, topdown=True):
            for file in files:
                # print(file)
                metapath = os.path.join(subdir, file)
                with open(metapath, 'r') as f:
                    meta = json.load(f)

                if "groundtruth" not in meta:
                    continue
                else:
                    if "helipad" in meta["groundtruth"]:
                        if not meta["groundtruth"]["helipad"]:
                            continue

                centers = self.get_center_box(meta)

                for center in centers:
                    if 300 <= center[0] <= 340 and 300 <= center[1] <= 340:
                        nb_helipads_centered += 1
                    nb_helipads += 1

        return [nb_helipads_centered/nb_helipads, nb_helipads_centered, nb_helipads]

    def get_center_shift(self, helipads_centers, image):
        height, width = image.shape[0], image.shape[1]
        image_center_x, image_center_y = height // 2, width // 2

        if len(helipads_centers) == 0:
            return []

        center_shifts = []

        for center in helipads_centers:
            x_c, y_c = center[0], center[1]

            x_shift = x_c - image_center_x
            y_shift = y_c - image_center_y

            print("({},{})<--({},{})+({},{})".format(x_c, y_c,
                                                     image_center_x,
                                                     image_center_y,
                                                     x_shift, y_shift))

            center_shifts.append((x_shift, y_shift))

        return center_shifts

    def draw_rectangle(self, image, meta):
        if "groundtruth" in meta:
            if "box" in meta["groundtruth"]:
                boxes = meta["groundtruth"]["box"]
                for box in boxes:
                    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255))
        return image

    def draw_centers(self, image, centers):
        for center in centers:
            cv2.circle(image, center, 5, (0, 0, 255))
        return image

    def get_zoom(self, meta):
        if "groundtruth" in meta:
            if "box" in meta["groundtruth"]:
                boxes = meta["groundtruth"]["box"]
                areas = []
                for box in boxes:
                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                    print("({},{}),({},{})".format(x1,y1,x2,y2))
                    # print("({},{})".format(x2-x1, y2-y1))
                    area = (x2-x1)*(y2-y1)
                    areas.append(abs(area)) # Absolute value
                print(areas)
                # return a list of zoom, sometimes plusieurs helipads
                return areas
            else:
                return meta["coordinates"]["zoom"]
        else:
            return meta["coordinates"]["zoom"]

    def center_helipads(self):

        for subdir, dirs, files in os.walk(self.meta_folder, topdown=True):
            for file in files:
                metapath = os.path.join(subdir, file)
                with open(metapath, 'r') as f:
                    meta = json.load(f)

                if "groundtruth" not in meta:
                    continue
                else:
                    if "helipad" in meta["groundtruth"]:
                        if not meta["groundtruth"]["helipad"]:
                            continue

                imagepath = os.path.join(self.image_folder, os.path.basename(subdir), os.path.splitext(file)[0] + ".png")
                image = cv2.imread(imagepath)

                helipads_centers = self.get_center_box(meta)

                center_shifts = self.get_center_shift(helipads_centers, image)

                zooms = self.get_zoom(meta)
                print("({},{})".format(meta["coordinates"]["latitude"],
                                       meta["coordinates"]["longitude"]))

                image = self.draw_rectangle(image, meta)
                image = self.draw_centers(image, helipads_centers)

                cv2.imshow('image', image)

                k = cv2.waitKey(0)



# get center shift
# get zoom shift

# get coordinates of images images, its center coordinates,
# and make relation between the shift and the shift coordinates
# problem if size of the image different ?

# 100696 --> +1 zoom
# 67797 --> +1 zoom
# WARNING : zoom 22 maybe no imagery !

# see the notebook

# after get augmented images


if __name__ == "__main__":

    meta_folder = os.path.join('C:\\', 'Users', 'jonas', 'Desktop', 'Helipad', 'Helipad_DataBase_meta', 'Helipad_DataBase_meta_original')
    image_folder = os.path.join('C:\\', 'Users', 'jonas', 'Desktop', 'Helipad', 'Helipad_DataBase', 'Helipad_DataBase_original')

    center_helipads = CenterHelipads(image_folder, meta_folder)

    # center_helipads.center_helipads()

    results = center_helipads.get_proportion_centered()

    print(results)

