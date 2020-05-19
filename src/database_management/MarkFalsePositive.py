import os
import json


class MarkFalsePositive:
    
    """
    Get the number of false positive and false negative predicted by a model. The detection has to be run with the model first. 
    """
    
    def __init__(self, meta_folder, model_number):
        """"
        `meta_folder`: the folder containing the meta file \n
        `model_numer`: the number of the model to check \n
        """
        self.meta_folder = meta_folder
        self.model_number = model_number

    def run(self):
        """
        Run the script.
        """
        nb_false_positive = 0
        nb_false_negative = 0

        for subdir, dirs, files in os.walk(self.meta_folder):
            for file in files:

                with open(os.path.join(subdir, file), 'r') as f:
                    meta = json.load(f)

                if "groundtruth" not in meta:
                    continue
                elif "predicted" not in meta:
                    continue
                elif "model_{}".format(self.model_number) not in meta["predicted"]:
                    continue

                groundtruth = meta["groundtruth"]
                predicted = meta["predicted"]["model_{}".format(self.model_number)]

                helipad_groundtruth = groundtruth["helipad"]
                helipad_predicted = predicted["helipad"]

                if not helipad_groundtruth and helipad_predicted:
                    nb_false_positive += 1
                if helipad_groundtruth and not helipad_predicted:
                    nb_false_negative += 1

        return nb_false_positive, nb_false_negative


if __name__ == "__main__":

    meta_folder = "C:\\Users\\jonas\\Desktop\\Helipad\\Helipad_DataBase_meta\\Helipad_DataBase_meta_original"
    model_number = 7

    mark_false_positive = MarkFalsePositive(meta_folder, model_number)

    nb_false_positive, nb_false_negative = mark_false_positive.run()

    print(nb_false_positive)
    print(nb_false_negative)






