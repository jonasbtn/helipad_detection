import os
import json

from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array


class BBPredict:

    def __init__(self, image_folder, meta_folder, model_number, model_path, tms=True):

        self.image_folder = image_folder
        self.model_path = model_path
        self.meta_folder = meta_folder
        self.model_number = model_number
        self.model = load_model(model_path)
        self.tms = tms

    def load_image(self, filename):
        image = load_img(filename, target_size=(32, 32))
        image = img_to_array(image)
        image = image.astype('float32')
        image = image*1.0/255.0
        return image

    def run(self):

        for subdir, dirs, files in os.walk(self.image_folder, topdown=True):
            for file in files:
                image_path = os.path.join(subdir, file)

                image = self.load_image(image_path)

                result = self.model.predict(image)[0]

                # 0 for false positive, 1 for helipad

                # save results of the prediction
                image_name = os.path.splitext(file)[0]
                image_info = image_name.split('_')
                zoom = image_info[1]
                xtile = image_info[2]
                ytile = image_info[3]
                image_id = image_info[4]

                meta_path = os.path.join(self.meta_folder,
                                         zoom,
                                         xtile,
                                         '_'.join(image_info[:len(image_info)-1]+'.meta'))

                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                f.close()

                predicted = meta["predicted"][f'model_{self.model_number}']

                nb_boxes = len(predicted["box"])

                if "cnn_validation" in predicted:
                    cnn_validation = predicted["cnn_validation"]
                else:
                    cnn_validation = [0 for i in range(nb_boxes)]

                cnn_validation[image_id] = result

                predicted["cnn_validation"] = cnn_validation

                meta["predicted"][f'model_{self.model_number}'] = predicted

                with open(meta_path, 'w') as f:
                    json.dump(meta, f, sort_keys=True, indent=4)
                f.close()


if __name__ == "__main__":

    image_folder =
    meta_folder =
    model_number =
    model_path =
    tms =

    bbpredict = BBPredict(image_folder=image_folder,
                          meta_folder=meta_folder,
                          model_number=model_number,
                          model_path=model_path,
                          tms=tms)

    bbpredict.run()









