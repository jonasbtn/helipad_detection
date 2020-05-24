import os
import json
import cv2
from tqdm import tqdm as tqdm

from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array


class BBPredict:
    
    """
    Run the prediction on the detected bounding boxes using the trained CNN from `BBTrainingManager`. \n
    The results are saved in the meta files of the original image as `cnn_validation` inside the prediction of the corresponding model.
    """ 
    
    def __init__(self, image_folder, meta_folder, model_number, model_path, tms=True,
                 index_path=None):
        """
        `image_folder`: string, path to the folder containing the images\n
        `meta_folder`: string, path to the folder containing the meta files \n
        `model_number`: int, number of the model which prediction has to be validated by the second CNN\n
        `model_path`: string, path to the model saved weights\n
        `tms`: boolean, True to indicate that the image folder follow TMS' structure\n
        `index_path`: string, path to the index files containing the names of the images that have bounding boxes inside them\n
        """
        self.image_folder = image_folder
        self.model_path = model_path
        self.meta_folder = meta_folder
        self.model_number = model_number
        self.model = load_model(model_path)
        self.tms = tms
        self.index_path = index_path
        if self.index_path:
            self.target_files = self.convert_meta_filename_to_path(image_folder, meta_folder, index_path)
        else:
            self.target_files = self.load_target_files()
        
    def load_target_files(self):
        """
        Load the target files.\n
        Returns a list of tuple (`image_path`, `meta_path`) 
        """
        target_files = []
        for subdir, dirs, files in os.walk(self.image_folder, topdown=True):
            for file in files:
                image_path = os.path.join(subdir, file)
                ytile = os.path.splitext(file)[0]
                xtile = os.path.basename(subdir)
                zoom = os.path.basename(os.path.dirname(subdir))
                meta_path = os.path.join(self.meta_folder,
                                         zoom,
                                         xtile,
                                         "Satellite_{}_{}_{}.meta".format(zoom,
                                                                          xtile,
                                                                          ytile))
                target_files.append([image_path, meta_path])
        return target_files
    
    @staticmethod
    def convert_meta_filename_to_path(image_folder, meta_folder, index_path):
        """
        From the index file, convert each  meta filename to a tuple (`image_path`, `meta_path`)\n
        `image_folder`: string, path to the root of the image folder\n
        `meta_folder`: string, path to the root of the meta_folder\n
        `index_path`: string, path to the index file
        """
        image_meta_path = []
        with open(index_path, 'r') as f:
            for meta_filename in f:
                info = meta_filename.split('_')
                ytile = info[3].split('.')[0]
                xtile = info[2]
                zoom = info[1]
                meta_path = os.path.join(meta_folder,
                                         zoom,
                                         xtile,
                                         "Satellite_{}_{}_{}.meta".format(zoom,
                                                                          xtile,
                                                                          ytile))
                image_path = os.path.join(image_folder,
                                          zoom,
                                          xtile, 
                                          ytile+".jpg")
                image_meta_path.append([image_path, meta_path])
        f.close()
        return image_meta_path

    def load_image(self, filename):
        """
        Load an image from `filename` and resize it to 64x64\n
        Returns the image
        """
        image = load_img(filename, target_size=(64, 64))
        image = img_to_array(image)
        image = image.astype('float32')
        image = image.reshape((1, 64, 64, 3))
        image = image*1.0/255.0
        return image

    def preprocess_image_box(self, image_box):
        """
        Preprocess the image bounding box by resizing it to 64x64\n
        Returns the image box resized.
        """
        image_box = cv2.resize(image_box, (64,64))
        image_box = img_to_array(image_box).astype('float32').reshape((1,64,64,3))
        image_box = image_box*1.0/255.0
        return image_box

    def run(self):
        """
        Run the prediction and save the results in the meta files.
        """
        for i in tqdm(range(len(self.target_files))):
            
            image_path = self.target_files[i][0]
            meta_path = self.target_files[i][1]
        
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            f.close()

            if not self.tms:
                if "groundtruth" not in meta:
                    continue
            if "predicted" not in meta:
                print("File not yet predicted by the model!")
                continue
            if "model_{}".format(self.model_number) not in meta["predicted"]:
                print("File not yet predicted by the model!")
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

            with open(meta_path, 'w') as f:
                json.dump(meta, f, sort_keys=True, indent=4)
            f.close()


if __name__ == "__main__":

    # image_folder = "C:\\Users\\AISG\\Documents\\Jonas\\Detected_Boxes\\model_7_0.0\\tms"
    # meta_folder = "C:\\Users\\AISG\\Documents\\Jonas\\Real_World_Dataset_TMS_meta_save_2\\Real_World_Dataset_TMS_meta\\sat"
    # model_number = 7
    # model_path = "final_model.h5"
    # tms = True

    image_folder = "C:\\Users\\AISG\\Documents\\Jonas\\Helipad\\Helipad_DataBase\\Helipad_DataBase_original"
    meta_folder = "C:\\Users\\AISG\\Documents\\Jonas\\Helipad\\Helipad_DataBase_meta\\Helipad_DataBase_meta_original"
    model_number = 7
    model_path = "final_model.h5"
    tms = False
    index_path = None

    bbpredict = BBPredict(image_folder=image_folder,
                          meta_folder=meta_folder,
                          model_number=model_number,
                          model_path=model_path,
                          tms=tms,
                          index_path=index_path)

    bbpredict.run()









