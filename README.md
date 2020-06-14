# HELIPADCAT: CATEGORISED HELIPAD IMAGE DATASET AND DETECTION METHOD

We present HelipadCat, a dataset of aerial images of helipads, together with a method to identify 
and locate such helipads from the air.

The jupyter notebook `helipad_detection_pipeline.ipynb` shows examples on how to execute every step of the method 
using our developed objects.

## 1. Dataset Creation 

Based on the FAA’s database of US airports, we create the first dataset of helipads, including 
a classification by visual helipad shape and features, which we make available to the research community. 
The dataset includes nearly 6,000 images with 12 different categories.

In the `data` folder is a csv file containing the dataset. The object `src.databse_management.csv_to_meta`
creates the annotated dataset and downloads the images from Google Maps.

Once the dataset created, we can start the next step.

## 2. Build Groundtruth

Annotate each image by drawing a bounding boxes around the helipads in the images.
This step has already been done.

## 3. Assign Categories

Assign a pattern category to each helipad. This step has already been done.

## 4. Dataset Augmentation

Apply Image augmentation on the images using Google's policy (`src.database_management.Database_augmentation`) or custom made policy with ImgAug (`src.database_management.database_augmentation_v2`)

## 5. Run Training

Run a training with `src.training.run_training` by specifying the dataset root folder, the model folder, tha augmented version, the train and test categories and the starting model weights.

## 6. Evaluate mAP on the original dataset

This step compute the mAP on the original dataset to get a first metric on the performances of the newly trained model.

## 7. Run Detection on Original Dataset to save bounding boxes

Here, with `src.detection.run_detection`, the images are feeded into the network and the bounding boxes 
are saved inside the meta files.

## 8. Run Benchmark

Once the bounding boxes have been saved, the accuracy, error, precision and recall are computed with `src.benchmark.run_benchmark`. A csv file is created inside the folder `src/benchmark/benchmark_results/` containing the results for a wide range of score threshold. 

## 9. Run Prediction on Satellite Images

In this step, with `src.detection.run_prediction_satellite`, additional unseen data are feeded into the network to detect helipads in any area of the world. 
First, the additional images have to be downloaded with SAS Planet software and store into a cache folder following the TMS file structure. 



