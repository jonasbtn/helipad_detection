import os
import json
import cv2

from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array


class BBPredict:

    def __init__(self, image_folder, meta_folder, model_number, model_path, tms=True,
                 extract_bounding_boxes=False):

        self.image_folder = image_folder
        self.model_path = model_path
        self.meta_folder = meta_folder
        self.model_number = model_number
        self.model = load_model(model_path)
        self.tms = tms
        self.extract_bounding_boxes = extract_bounding_boxes

    def load_image(self, filename):
        image = load_img(filename, target_size=(32, 32))
        image = img_to_array(image)
        image = image.astype('float32')
        image = image.reshape((1, 32, 32, 3))
        image = image*1.0/255.0
        return image

    def preprocess_image_box(self, image_box):
        image_box = cv2.resize(image_box, (32,32))
        image_box = img_to_array(image_box).astype('float32').reshape((1,32,32,3))
        image_box = image_box*1.0/255.0
        return image_box

    def run(self):

        for subdir, dirs, files in os.walk(self.image_folder, topdown=True):

            print(os.path.basename(subdir))

            for file in files:
                image_path = os.path.join(subdir, file)

                if self.tms:
                    ytile = os.path.splitext(file)[0]
                    xtile = os.path.basename(subdir)
                    zoom = os.path.basename(os.path.dirname(subdir))
                    meta_path = os.path.join(self.meta_folder,
                                             zoom,
                                             xtile,
                                             "Satellite_{}_{}_{}.meta".format(zoom,
                                                                              xtile,
                                                                              ytile))
                else:
                    meta_path = os.path.join(self.meta_folder,
                                             os.path.basename(subdir),
                                             os.path.splitext(file)[0] + ".meta")

                if not os.path.isfile(meta_path):
                    meta = dict()
                else:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    f.close()

                if not self.tms:

                    if "groundtruth" not in meta:
                        continue
                    elif "predicted" not in meta:
                        continue
                    elif "model_{}".format(self.model_number) not in meta["predicted"]:
                        continue

                    image = cv2.imread(image_path)

                    predicted = meta["predicted"]["model_{}".format(self.model_number)]

                    predicted_boxes = predicted["box"]

                    if "cnn_validation" in predicted:
                        cnn_validation = predicted["cnn_validation"]
                    else:
                        cnn_validation = [0 for i in range(len(predicted_boxes))]

                    for i in range(len(predicted_boxes)):
                        box = predicted_boxes[i]
                        x_min = min(box[0], box[2])
                        y_min = min(box[1], box[3])
                        x_max = max(box[2], box[0])
                        y_max = max(box[3], box[1])
                        image_box = image[y_min:y_max, x_min:x_max, :]
                        image_box = self.preprocess_image_box(image_box)

                        result = float(self.model.predict(image_box)[0][0])

                        cnn_validation[i] = result

                    predicted["cnn_validation"] = cnn_validation

                    meta["predicted"][f'model_{self.model_number}'] = predicted

                else:

                    image = self.load_image(image_path)

                    result = float(self.model.predict(image)[0][0])
                    print(result)
                    # if result <= 0.5:
                    #     result = 0
                    # else:
                    #     result = 1
                    # 0 for false positive, 1 for helipad
                    # print(image_path)
                    # print(result)
                    # save results of the prediction
                    image_name = os.path.splitext(file)[0]
                    image_info = image_name.split('_')
                    zoom = image_info[1]
                    xtile = image_info[2]
                    ytile = image_info[3]
                    image_id = int(image_info[4])
                    # print(image_info)
                    # print('_'.join(image_info[:len(image_info)-1])+'.meta')
                    meta_path = os.path.join(self.meta_folder,
                                             zoom,
                                             xtile,
                                             '_'.join(image_info[:len(image_info)-1])+'.meta')
                    print(meta_path)
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    f.close()
                    print(meta)
                    predicted = meta["predicted"][f'model_{self.model_number}']

                    nb_boxes = len(predicted["box"])

                    if "cnn_validation" in predicted:
                        cnn_validation = predicted["cnn_validation"]
                    else:
                        cnn_validation = [0 for i in range(nb_boxes)]

                    cnn_validation[image_id] = result

                    predicted["cnn_validation"] = cnn_validation

                    meta["predicted"][f'model_{self.model_number}'] = predicted
                    print(meta)

                with open(meta_path, 'w') as f:
                    json.dump(meta, f, sort_keys=True, indent=4)
                f.close()


if __name__ == "__main__":

    # image_folder = "C:\\Users\\AISG\\Documents\\Jonas\\Detected_Boxes\\model_7_0.0\\tms"
    # meta_folder = "C:\\Users\\AISG\\Documents\\Jonas\\Real_World_Dataset_TMS_meta_save_2\\Real_World_Dataset_TMS_meta\\sat"
    # model_number = 7
    # model_path = "final_model.h5"
    # tms = True
    # extract_bounding_boxes = False

    image_folder = "C:\\Users\\AISG\\Documents\\Jonas\\Helipad\\Helipad_DataBase\\Helipad_DataBase_original"
    meta_folder = "C:\\Users\\AISG\\Documents\\Jonas\\Helipad\\Helipad_DataBase_meta\\Helipad_DataBase_meta_original"
    model_number = 7
    model_path = "final_model.h5"
    tms = False
    extract_bounding_boxes = True

    bbpredict = BBPredict(image_folder=image_folder,
                          meta_folder=meta_folder,
                          model_number=model_number,
                          model_path=model_path,
                          tms=tms,
                          extract_bounding_boxes=extract_bounding_boxes)

    bbpredict.run()









