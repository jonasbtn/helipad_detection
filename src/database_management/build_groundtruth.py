import os
import matplotlib.pyplot as plt
import cv2
import shutil
import json


class GroundTruth:

    def __init__(self, database_folder, meta_folder, review=False,
                 augment_only=False, redo_false=False, redo_true=False,
                 tms_dataset=False):
        self.database_folder = database_folder
        self.meta_folder = meta_folder
        self.target_files = self.build_target_files(review, augment_only, redo_false, redo_true, tms_dataset)

        print("%d more files to go !" % len(self.target_files))

    def build_target_files(self, review, augment_only, redo_false, redo_true, tms_dataset):
        target_files = []
        for subdir, dirs, files in os.walk(self.meta_folder, topdown=True):

            if augment_only:
                if not os.path.basename(subdir)[7:16] == 'augmented':
                    continue

            for file in files:
                metapath = os.path.join(subdir, file)

                if not tms_dataset:
                    filepath = os.path.join(self.database_folder,
                                            os.path.basename(subdir),
                                            os.path.splitext(file)[0] + ".png")
                else:
                    zoom_level = os.path.basename(os.path.dirname(subdir))
                    xtile = os.path.basename(subdir)
                    ytile = os.path.splitext(file)[0].split('_')[3] + ".jpg"
                    filepath = os.path.join(self.database_folder,
                                            zoom_level,
                                            xtile,
                                            ytile)
                    print(filepath)

                with open(metapath, 'r') as f:
                    meta = json.load(f)
                f.close()
                if "groundtruth" in meta:
                    if "helipad" in meta["groundtruth"]:
                        if meta["groundtruth"]["helipad"]:
                            if "box" not in meta["groundtruth"] or len(meta["groundtruth"]["box"]) == 0 or review or redo_true:
                                if os.path.isfile(filepath):
                                    target = [filepath, metapath]
                                    target_files.append(target)
                                    print("Added : " + metapath)
                    else:
                        if redo_false:
                            if os.path.isfile(filepath):
                                target = [filepath, metapath]
                                target_files.append(target)
                                print("Added : " + metapath)
                else:
                    meta["groundtruth"] = {}
                    target = [filepath, metapath]
                    target_files.append(target)
                    with open(metapath, 'w') as f:
                        json.dump(meta, f, indent=4, sort_keys=True)
                    f.close()
        return target_files

    def shape_selection(self, event, x, y, flags, param):
        # # grab references to the global variables
        # global ref_point, crop

        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being performed
        if event == cv2.EVENT_LBUTTONDOWN:
            self.ref_point = [(x, y)]

        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # the cropping operation is finished
            self.ref_point.append((x, y))

            print(self.ref_point)

            self.boxes.append([self.ref_point[0][0], self.ref_point[0][1],
                               self.ref_point[1][0], self.ref_point[1][1]])

            # draw a rectangle around the region of interest
            cv2.rectangle(self.image, self.ref_point[0], self.ref_point[1], (0, 0, 255), 2)
            cv2.imshow("image", self.image)

    def run(self):

        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.shape_selection)

        i = 0

        while i < len(self.target_files):

            target_meta = self.target_files[i]

            filepath = target_meta[0]
            metapath = target_meta[1]

            with open(metapath, 'r') as f:
                meta = json.load(f)

            if "box" not in meta["groundtruth"]:
                self.boxes = []
            else:
                self.boxes = meta["groundtruth"]["box"]

            self.image = cv2.imread(filepath)
            clone = self.image.copy()

            self.ref_point = []

            for box in self.boxes:
                cv2.rectangle(self.image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

            cv2.imshow('image', self.image)

            # draw rectangle

            # clic on the rectangle + e to remove it

            # press s to save
            # we will do the categories later

            # press backspace to come back

            k = cv2.waitKey(0)

            if k == ord("s"):
                if len(self.boxes) > 0:
                    meta["groundtruth"]["helipad"] = True
                    meta["groundtruth"]["box"] = self.boxes
                else:
                    meta["groundtruth"]["helipad"] = False
                    meta["groundtruth"]["box"] = self.boxes
                with open(metapath, 'w') as f:
                    json.dump(meta, f, indent=4, sort_keys=True)
            elif k == 8:
                i = i - 1
                continue
            # press 'r' to reset the window
            elif k == ord("r"):
                self.image = clone.copy()
                self.boxes = []
                meta["groundtruth"]["helipad"] = False
                if "box" in meta["groundtruth"]:
                    meta["groundtruth"]["box"] = []
                with open(metapath, 'w') as f:
                    json.dump(meta, f, indent=4, sort_keys=True)
                continue
            # if the 'q' key is pressed, break from the loop
            elif k == ord("q"):
                break

            i = i + 1

            print("{} more to go!".format(len(self.target_files) - i))

        # close all open windows
        cv2.destroyAllWindows()


if __name__ == "__main__":

    database_folder = "C:\\Users\\jonas\\Desktop\\Detection\\Detection_Dataset"
    meta_folder = "C:\\Users\\jonas\\Desktop\\Detection\\Detection_Dataset_meta"

    # meta_folder = os.path.join('C:\\', 'Users', 'jonas', 'Desktop', 'Helipad', 'Helipad_DataBase_meta', 'Helipad_DataBase_meta_original')
    # database_folder = os.path.join('C:\\', 'Users', 'jonas', 'Desktop', 'Helipad', 'Helipad_DataBase', 'Helipad_DataBase_original')

    ground_truth = GroundTruth(database_folder,
                               meta_folder,
                               review=False,
                               augment_only=False,
                               redo_false=False,
                               redo_true=True,
                               tms_dataset=True)

    ground_truth.run()