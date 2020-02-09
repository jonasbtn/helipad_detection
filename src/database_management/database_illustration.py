import os
import json
import numpy as np
import cv2


class DatabaseIllustration:

    def __init__(self, image_folder, meta_folder):
        self.image_folder = image_folder
        self.meta_folder = meta_folder


    def display_grid_groundtruth(self, width, height):

        nb_images = width * height
        grid_image = np.zeros((640*height, 640*width, 3))
        added_images = 0

        for subdir, dirs, files in os.walk(self.image_folder, topdown=True):
            for file in files:
                if file[0] == ".":
                    continue
                image_path = os.path.join(subdir, file)
                meta_filepath = os.path.join(self.meta_folder,
                                             os.path.basename(subdir),
                                             os.path.splitext(file)[0] + ".meta")
                if added_images >= nb_images:
                    break

                with open(meta_filepath, 'r') as f:
                    meta = json.load(f)
                if not "groundtruth" in meta:
                    continue
                elif not meta["groundtruth"]["helipad"]:
                    continue
                # or add the false positive here as not helipad ?
                elif "box" not in meta["groundtruth"]:
                    continue

                bboxes = meta["groundtruth"]["box"]
                image = cv2.imread(image_path)
                for box in bboxes:
                    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

                i = added_images // width
                j = added_images % width

                print(image.shape)
                print(grid_image.shape)

                grid_image[i*640:(i+1)*640, j*640:(j+1)*640, :] = image
                added_images += 1
                print(added_images)

        # cv2.imshow('image', grid_image)
        #
        # cv2.waitKey(0)

        cv2.imwrite('groundtruth.jpg', grid_image)

    def display_grid_categories(self, width, height):

        nb_images = width * height
        grid_image = np.zeros((640*height, 640*width, 3))
        added_images = 0

        added_categories = {}
        for i in range(10):
            added_categories[str(i)] = 0
        added_categories["d"] = 0
        added_categories["u"] = 0

        for subdir, dirs, files in os.walk(self.image_folder, topdown=True):
            for file in files:
                if file[0] == ".":
                    continue
                image_path = os.path.join(subdir, file)
                meta_filepath = os.path.join(self.meta_folder,
                                             os.path.basename(subdir),
                                             os.path.splitext(file)[0] + ".meta")
                # image_name = os.path.splitext(file)[0]
                # image_number = int(image_name.split('_')[1])

                # if image_number <= 4250:
                #     continue

                if added_images >= nb_images:
                    break

                with open(meta_filepath, 'r') as f:
                    meta = json.load(f)
                if not "groundtruth" in meta:
                    continue
                elif not meta["groundtruth"]["helipad"]:
                    continue
                # or add the false positive here as not helipad ?
                elif "box" not in meta["groundtruth"]:
                    continue
                elif "category" not in meta["groundtruth"]:
                    continue

                if "category" not in meta["groundtruth"]:
                    continue

                category = meta["groundtruth"]["category"]

                if category == 'u':
                    print(file)

                if added_categories[category] < 25:
                    added_categories[category] += 1
                    continue
                if added_categories[category] > 25:
                    continue

                bboxes = meta["groundtruth"]["box"]
                image = cv2.imread(image_path)
                # image = image / 256

                model_number = 5
                predicted = meta["predicted"]["model_{}".format(model_number)]
                bboxes_pred = predicted["box"]
                scores_pred = predicted["score"]

                for box in bboxes:
                    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                    if category=="d":
                        category="O"
                    elif category=="u":
                        category="U"
                    cv2.putText(image, "Category : {}".format(category),
                                (170, 620),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.5,
                                (0, 0, 255),
                                3,
                                cv2.LINE_AA)
                    if category=="O":
                        category="d"
                    elif category=="U":
                        category="u"
                for i in range(len(bboxes_pred)):
                    box = bboxes_pred[i]
                    score = scores_pred[i]
                    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                    cv2.putText(image, "{}:{}".format(model_number, str(score)[:4]),
                                (box[0] + 10, box[1]),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255, 0, 0),
                                2,
                                cv2.LINE_AA)

                if "0" <= category <= "9":
                    int_cat = int(category)
                elif category == "d":
                    int_cat = 10
                elif category == "u":
                    int_cat = 11

                i = int_cat // width
                j = int_cat % width

                print(image.shape)
                print(grid_image.shape)

                grid_image[i*640:(i+1)*640, j*640:(j+1)*640, :] = image
                added_images += 1
                print(added_images)

                added_categories[category] += 1

        cv2.imwrite('categories_with_pred_5.jpg', grid_image)

        print(added_categories)

    def categories_stats(self):
        added_categories = {}
        for i in range(10):
            added_categories[str(i)] = 0
        added_categories["d"] = 0
        added_categories["u"] = 0

        for subdir, dirs, files in os.walk(self.image_folder, topdown=True):
            for file in files:
                if file[0] == ".":
                    continue
                image_path = os.path.join(subdir, file)
                meta_filepath = os.path.join(self.meta_folder,
                                             os.path.basename(subdir),
                                             os.path.splitext(file)[0] + ".meta")
                image_name = os.path.splitext(file)[0]
                image_number = int(image_name.split('_')[1])

                if image_number > 4250:
                    continue

                with open(meta_filepath, 'r') as f:
                    meta = json.load(f)
                if not "groundtruth" in meta:
                    continue
                elif not meta["groundtruth"]["helipad"]:
                    continue
                # or add the false positive here as not helipad ?
                elif "box" not in meta["groundtruth"]:
                    continue
                elif "category" not in meta["groundtruth"]:
                    continue

                if "category" not in meta["groundtruth"]:
                    continue

                category = meta["groundtruth"]["category"]

                added_categories[category] += 1

        print(added_categories)


if __name__ == "__main__":

    image_folder = os.path.join('C:\\', 'Users', 'jonas', 'Desktop', 'Helipad', 'Helipad_DataBase', 'Helipad_DataBase_original')
    meta_folder = os.path.join('C:\\', 'Users', 'jonas', 'Desktop', 'Helipad', 'Helipad_DataBase_meta', 'Helipad_DataBase_meta_original')

    database_illustration = DatabaseIllustration(image_folder, meta_folder)

    # database_illustration.display_grid_groundtruth(4,3)

    # database_illustration.display_grid_categories(2,6)

    database_illustration.categories_stats()
