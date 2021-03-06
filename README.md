# HELIPADCAT: CATEGORISED HELIPAD IMAGE DATASET AND DETECTION METHOD

We present HelipadCat, a dataset of aerial images of helipads, together with a method to identify and locate such helipads from the air.
Please cite the following paper if you use our data or code:

>	J. Bitoun, S. Winkler. "HelipadCat: Categorised helipad image dataset and detection method." 
> In Proc. IEEE Region 10 International Conference (TENCON), Osaka, Japan, Nov. 16-19, 2020.

The jupyter notebook `helipad_detection_pipeline.ipynb` shows examples on how to execute every step of the method 
using our developed objects.

## 1. Dataset Creation 

Based on the FAA’s database of US airports, we create the first dataset of helipads, including 
a classification by visual helipad shape and features, which we make available to the research community. 
The dataset includes nearly 6,000 images with 12 different categories.

In the `data` folder is a csv file containing the dataset. The object `src.databse_management.csv_to_meta`
creates the annotated dataset and downloads the images from Google Maps.

![alt text](https://github.com/jonasbtn/helipad_detection/blob/master/data/HelipadExample.PNG?raw=true)

Once the dataset created, we can start the next step.

## 2. Build Groundtruth

Annotate each image by drawing a bounding boxes around the helipads in the images. This information is already inside the csv files.

![alt text](https://github.com/jonasbtn/helipad_detection/blob/master/data/HelipadGroundtruthExample.PNG?raw=true)

## 3. Assign Categories

Assign a pattern category to each helipad. This information is already inside the csv files.

![alt text](https://github.com/jonasbtn/helipad_detection/blob/master/data/HelipadCategories.PNG?raw=true)

## 4. Dataset Augmentation

Apply Image augmentation on the images using Google's policy (`src.database_management.Database_augmentation`) or custom made policy with ImgAug (`src.database_management.database_augmentation_v2`)

![alt text](https://github.com/jonasbtn/helipad_detection/blob/master/data/Aug_Illustration.PNG?raw=true)

## 5. Run Training

Run a training with `src.training.run_training` by specifying the dataset root folder, the model folder, tha augmented version, the train and test categories and the starting model weights.

The training framework MRCNN has to be installed. See : <https://github.com/matterport/Mask_RCNN>

Once the training has started, it is possible to keep track of the evolution of the metrics with Tensorboard. Simply launch Tensorboard with the command : `tensorboard --logdir model_dir_path --port 8890`.

![alt text](https://github.com/jonasbtn/helipad_detection/blob/master/data/loss_model4.PNG?raw=true)

## 6. Evaluate mAP on the original dataset

This step compute the mAP on the original dataset to get a first metric on the performances of the newly trained model. The object used is the same as in the previous step. The method to execute is : `RunTraining.run_predict()`. 

## 7. Run Detection on Original Dataset to save bounding boxes

Here, with `src.detection.run_detection`, the images are feeded into the network and the bounding boxes 
are saved inside the meta files. It is then possible to compare the model prediction with the groundtruth on the dataset :

![alt text](https://github.com/jonasbtn/helipad_detection/blob/master/data/Prediction_illustration.PNG?raw=true)

## 8. Run Benchmark

Once the bounding boxes have been saved, the accuracy, error, precision and recall are computed with `src.benchmark.run_benchmark`. A csv file is created inside the folder `src/benchmark/benchmark_results/` containing the results for a wide range of score threshold. Below is an example of benchmark with different score threshold : 

![alt text](https://github.com/jonasbtn/helipad_detection/blob/master/data/Benchmark_example.PNG?raw=true)

## 9. Run Prediction on Satellite Images

In this step, with `src.detection.run_prediction_satellite`, additional unseen data are feeded into the network to detect helipads in any area of the world. 
First, the additional images have to be downloaded with SAS Planet software and store into a cache folder following the TMS file structure, preferably with a zoom of 19. 
The bounding boxes are saved into meta files. The predictions will be vizualised on the map after our custom filtering. 

## 10. Build index path by score on Satellite Images

When the area to scan is big, the number of images to feed into the network is high. After the detection, since the number of helipads is very low compared to the number of images, a text file is created, serving as an index file, having in each line the location of the meta file with a bounding box inside. The object `src.database_management.index_path_by_score` allows such indexation.

## 11. Optional Step : Train a KNN model to serve as a validation model

The idea behind this step is to have a KNN who validates the detected bounding boxes as true or false. This approach did not give good performance.

## 12. Optional Step : Train a CNN model to serve as a validation model

The idea behind this step is to have another network who validates the detected bounding boxes as true or false. This approach did not give good performance.

## 13. Area Computation

Our first custom filter computes the ground area of the bounding boxes in meter squared. The areas are saved into the meta files. The object used is `src.bb_analysis.bb_compute_area_tms`.

## 14. Apply Shadow Detection

Our second custom filter looks for the presence of shadow inside the bounding boxes. The object used is `src.bb_analysis.run_shadow_detection`. The results are saved into the meta files. Below are examples of the shadow detections on helipads and on false positives : 

![alt text](https://github.com/jonasbtn/helipad_detection/blob/master/data/ShadowIllustration.PNG?raw=true)

## 15. Optional Step : Build groundtruth on Satellite Images

This step allows the user to manually annotated the groundtruth of the detected bounding boxes on additional data in order to compute the performance of the model. The object is `src.database_management.build_groundtruth_tms`. With the index file created in step 10, only the images with a bounding boxes are to be validated manually as true or false. 

## 16. Benchmark on Satellite images annotated

After step 15, it is then possible to compute the benchmark on additional data with the object `src.benchmark.benchmark_manager_TMS`. The user can configure the parameters of our three custom filters (shadow, area and score). 

![alt text](https://github.com/jonasbtn/helipad_detection/blob/master/data/Benchmark_filteres_example.PNG?raw=true)

## 17. Build Placemarks

This final step of our method creates a placemarks file in order to visualize the detection on the map. Each center of the bounding boxes are pin points on the map of the world, as shown below. The object used is `src.detection.build_placemarks`. 


Unfiltered Detection of Model F:

![alt text](https://github.com/jonasbtn/helipad_detection/blob/master/data/Placemarks_Model_7.PNG?raw=true)

Filtered Detection of Model F:

![alt text](https://github.com/jonasbtn/helipad_detection/blob/master/data/Placemarks_Model_7_filtered.PNG?raw=true)

Closer look at the placemarks:

<img src="https://github.com/jonasbtn/helipad_detection/blob/master/data/HelipadDetected_1.PNG?raw=true" width="425"/> <img src="https://github.com/jonasbtn/helipad_detection/blob/master/data/HelipadDetected_2.PNG?raw=true" width="425"/>
