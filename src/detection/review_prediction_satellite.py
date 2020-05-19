import os
import cv2
import json


class ReviewPredictionSatellite:
    
    """
    Review the prediction of the satellite images by displaying them 
    """
    
    def __init__(self, cache_tms_sat_folder, meta_folder, zoom_level, model_number, helipad_only=True):
        self.cache_tms_folder = cache_tms_sat_folder
        self.image_folder = os.path.join(self.cache_tms_folder, str(zoom_level))
        self.meta_folder = meta_folder
        self.meta_folder = os.path.join(self.meta_folder, str(zoom_level))
        self.zoom_level = zoom_level
        self.model_number = model_number
        print('Loading Files')
        self.target_files = self.build_target_files(helipad_only=helipad_only)
        print("{} files loaded".format(str(len(self.target_files))))

    def build_target_files(self, helipad_only=True):
        target_files = []
        for subdir, dirs, files in os.walk(self.meta_folder):
            for file in files:
                xtile = os.path.basename(subdir)
                file_info = os.path.splitext(file)[0]
                file_info = file_info.split('_')
                if len(file_info)<4:
                    print(file)
                ytile = int(file_info[3])

                # ytile = os.path.splitext(file)[0]
                meta_filepath = os.path.join(subdir, file)
                filepath = os.path.join(self.image_folder,
                                             str(xtile),
                                             str(ytile)+".jpg")
                if not os.path.isfile(filepath):
                    print("File doesnt exit: {}".format(filepath))
                    continue
                if helipad_only:
                    with open(meta_filepath, 'r') as f:
                        meta = json.load(f)

                    if "predicted" not in meta:
                        print("no predicted")
                        continue
                    key = "model_{}".format(self.model_number)
                    if key not in meta["predicted"]:
                        print("model not predicted")
                        continue
                    if "helipad" not in meta["predicted"][key]:
                        print("no helipad")
                        continue
                    if not meta["predicted"][key]["helipad"]:
                        print("its not an helipad")
                        continue

                target_files.append([filepath, meta_filepath])

        return target_files

    def review_prediction(self):

        predict_color = (255, 0, 0)

        for target_path in self.target_files:
            image_path = target_path[0]
            meta_path = target_path[1]
            image = cv2.imread(image_path)
            print(image_path)
            with open(meta_path, 'r') as f:
                meta = json.load(f)

            if "predicted" in meta:
                predicted = meta["predicted"]
                key = "model_{}".format(self.model_number)
                if key in predicted:
                    model_prediction = predicted[key]
                    bboxes = model_prediction["box"]
                    scores = model_prediction["score"]
                    for i in range(len(bboxes)):
                        box = bboxes[i]
                        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), predict_color, 2)
                        cv2.putText(image, "{}:{}".format(self.model_number, str(scores[i])),
                                    (box[0] + 10, box[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    predict_color,
                                    2,
                                    cv2.LINE_AA)
                        print(box)
                        print(predicted[key]["coordinates"]["center"][i])

            cv2.imshow('image', image)
            k = cv2.waitKey(0)


if __name__ == "__main__":
    # cache_tms_sat_folder = "C:\\Users\\jonas\\Desktop\\SAS.Planet.Release.191221\\cache_tms\\sat"
    #
    # cache_tms_sat_folder = "C:\\Users\\jonas\\Desktop\\Real_World_Test_DataBase"
    # meta_folder = "C:\\Users\\jonas\\Desktop\\Real_World_Test_DataBase_meta"

    cache_tms_sat_folder = "C:\\Users\\jonas\\Desktop\\Detection\\Detection_Dataset"
    meta_folder = "C:\\Users\\jonas\\Desktop\\Detection\\Detection_Dataset_meta"

    zoom_level = 18
    model_number = 8

    review_prediction_satellite = ReviewPredictionSatellite(cache_tms_sat_folder=cache_tms_sat_folder,
                                                            meta_folder=meta_folder,
                                                            zoom_level=zoom_level,
                                                            model_number=model_number,
                                                            helipad_only=True)

    review_prediction_satellite.review_prediction()


