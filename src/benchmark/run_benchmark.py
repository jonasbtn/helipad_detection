import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn import metrics
import pathlib

from helipad_detection.src.benchmark.benchmark_manager import BenchmarkManager


class RunBenchmark:
    
    """
    Run a multiple benchmarks on a specific dataset with wide range of score threshold. \n
    The results are saved in a csv file. The user can then find the optimal score threshold that fit best the dataset. 
    """

    def __init__(self, image_folder, meta_folder, model_numbers,
                 test_only=True, tms_dataset=False, zoom_level=None,
                 include_category=None,
                 include_negative=True,
                 city_lat=None,
                 train_only=False):

        self.image_folder = image_folder
        self.meta_folder = meta_folder
        self.model_numbers = model_numbers
        self.tms_dataset = tms_dataset
        self.zoom_level = zoom_level
        self.city_lat = city_lat
        self.test_only = test_only
        self.include_category = include_category
        self.include_negative = include_negative
        self.train_only = train_only

        self.benchmark_manager = BenchmarkManager(image_folder,
                                                  meta_folder,
                                                  test_only=test_only,
                                                  tms_dataset=tms_dataset,
                                                  zoom_level=zoom_level,
                                                  include_category=include_category,
                                                  include_negative=include_negative,
                                                  city_lat=city_lat,
                                                  train_only=train_only)

    def run(self, threshold_validation=None):
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
                                                 threshold_area,
                                                 threshold_validation=threshold_validation)
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
                if threshold_validation:
                    filename = "benchmark_model_{}_t{}_test{}_train{}.csv".format(model_number, str(threshold_validation), self.test_only, self.train_only)
                if self.include_category:
                    filename = "benchmark_model_{}_c{}_n{}_test{}_train{}.csv".format(model_number, "".join(self.include_category), self.include_negative, self.test_only, self.train_only)
                else:
                    filename = "benchmark_model_{}_{}.csv".format(model_number, self.test_only)
            df.to_csv(os.path.join(pathlib.Path(__file__).parent.absolute(), "benchmark_results", filename))

        df_auc = pd.DataFrame(data=aucs, columns=["Model Number", "AUC"])

        df_auc.to_csv("benchmark_models_auc.csv")


if __name__ == "__main__":

    # image_folder = "C:\\Users\\jonas\\Desktop\\Helipad\\Helipad_DataBase\\Helipad_DataBase_original"
    # meta_folder = "C:\\Users\\jonas\\Desktop\\Helipad\\Helipad_DataBase_meta\\Helipad_DataBase_meta_original"

    image_folder = "C:\\Users\\AISG\\Documents\\Jonas\\Helipad\\Helipad_DataBase\\Helipad_DataBase_original"
    meta_folder = "C:\\Users\\AISG\\Documents\\Jonas\\Helipad\\Helipad_DataBase_meta\\Helipad_DataBase_meta_original"

    # image_folder = "C:\\Users\\jonas\\Desktop\\Real_World_Test_DataBase"
    # meta_folder = "C:\\Users\\jonas\\Desktop\\Real_World_Test_DataBase_meta"

    # image_folder = "../../Helipad_DataBase/Helipad_DataBase_original"
    # meta_folder = "../../Helipad_DataBase_meta/Helipad_DataBase_meta_original"

    # test_only = True
    # include_category = ["1", "2", "3", "5", "6", "8", "9", "d"]
    # tms_dataset = False
    # zoom_level = None

    # image_folder = "../../../Detection/Detection_Dataset/"
    # meta_folder = "../../../Detection/Detection_Dataset_meta/"
    #
    model_numbers = [7]

    run_benchmark = RunBenchmark(image_folder,
                                 meta_folder,
                                 model_numbers)

    run_benchmark.run(threshold_validation=0.99)


    # test_only = False
    # include_category = None
    # tms_dataset = False
    # zoom_level = None

    # cities_lat = [['los_angeles', '44'],
    #             ['paris', '13'],
    #             ['manille', '21'],
    #             ['tokyo', '23']]
    #
    # for city_lat in cities_lat:
    #     print(city_lat),
    #     run_benchmark = RunBenchmark(image_folder,
    #                                  meta_folder,
    #                                  model_numbers,
    #                                  test_only=test_only,
    #                                  tms_dataset=tms_dataset,
    #                                  zoom_level=zoom_level,
    #                                  include_category=include_category,
    #                                  city_lat=city_lat)
    #
    #     run_benchmark.run()







