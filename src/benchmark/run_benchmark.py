import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn import metrics

from benchmark_manager import BenchmarkManager


class RunBenchmark:

    def __init__(self, image_folder, meta_folder, model_numbers,
                 test_only=True, tms_dataset=False, zoom_level=None,
                 include_category = None,
                 city_lat = None):

        self.image_folder = image_folder
        self.meta_folder = meta_folder
        self.model_numbers = model_numbers
        self.tms_dataset = tms_dataset
        self.zoom_level = zoom_level
        self.city_lat = city_lat

        self.benchmark_manager = BenchmarkManager(image_folder,
                                                  meta_folder,
                                                  test_only=test_only,
                                                  tms_dataset=tms_dataset,
                                                  zoom_level=zoom_level,
                                                  include_category=include_category,
                                                  city_lat=city_lat)

    def run(self):
        aucs = []
        for model_number in self.model_numbers:

            print("Benchmarking model {}".format(model_number))
            results = []
            threshold_iou = 0.5
            threshold_area = 0.8

            threshold_scores = []
            threshold_scores.extend(list(np.arange(0.0, 0.6, 0.1)))
            threshold_scores.extend(list(np.arange(0.6, 0.8, 0.01)))
            # threshold_scores.extend(list(np.arange(0.9, 0.97, 0.01)))
            threshold_scores.extend(list(np.arange(0.8, 0.99, 0.001)))
            threshold_scores.extend(list(np.arange(0.99, 0.9999, 0.0001)))

            threshold_scores = list(np.array(threshold_scores))

            for i in tqdm(range(len(threshold_scores))):
                threshold_score = threshold_scores[i]
                res = self.benchmark_manager.run(model_number,
                                                 threshold_score,
                                                 threshold_iou,
                                                 threshold_area)
                results.append(res)

            df = pd.DataFrame(data=results,
                              columns=["Model Number", "Threshold Score", "Threshold IOU", "Threshold Area",
                                       "Accuracy", "Error", "Precision", "Recall", "FPR", "TPR",
                                       "TP", "TN", "FP", "FN"])

            auc = metrics.auc(df["FPR"].values, df["TPR"].values)

            aucs.append([model_number, auc])

            if self.tms_dataset:
                if self.city_lat:
                    filename = "benchmark_model_{}_tms_z{}_{}.csv".format(model_number,
                                                                          self.zoom_level,
                                                                          self.city_lat[0])
                else:
                    filename = "benchmark_model_{}_tms_z{}.csv".format(model_number, self.zoom_level)
            else:
                filename = "benchmark_model_{}.csv".format(model_number)
            df.to_csv(filename)

        df_auc = pd.DataFrame(data=aucs, columns=["Model Number", "AUC"])

        df_auc.to_csv("benchmark_models_auc.csv")


if __name__ == "__main__":

    # image_folder = "C:\\Users\\jonas\\Desktop\\Helipad\\Helipad_DataBase\\Helipad_DataBase_original"
    # meta_folder = "C:\\Users\\jonas\\Desktop\\Helipad\\Helipad_DataBase_meta\\Helipad_DataBase_meta_original"

    # image_folder = "C:\\Users\\jonas\\Desktop\\Real_World_Test_DataBase"
    # meta_folder = "C:\\Users\\jonas\\Desktop\\Real_World_Test_DataBase_meta"

    # image_folder = "../../Helipad_DataBase/Helipad_DataBase_original"
    # meta_folder = "../../Helipad_DataBase_meta/Helipad_DataBase_meta_original"

    # test_only = True
    # include_category = ["1", "2", "3", "5", "6", "8", "9", "d"]
    # tms_dataset = False
    # zoom_level = None

    image_folder = "../../../Detection/Detection_Dataset/"
    meta_folder = "../../../Detection/Detection_Dataset_meta/"

    test_only = False
    include_category = None
    tms_dataset = True
    zoom_level = 18

    model_numbers = [4]

    cities_lat = [['los_angeles', '44'],
                ['paris', '13'],
                ['manille', '21'],
                ['tokyo', '23']]

    for city_lat in cities_lat:
        print(city_lat)
        run_benchmark = RunBenchmark(image_folder,
                                     meta_folder,
                                     model_numbers,
                                     test_only=test_only,
                                     tms_dataset=tms_dataset,
                                     zoom_level=zoom_level,
                                     include_category=include_category,
                                     city_lat=city_lat)

        run_benchmark.run()







