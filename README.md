# HELIPADCAT: CATEGORISED HELIPAD IMAGE DATASET AND DETECTION METHOD

We present HelipadCat, a dataset of aerial images of helipads, together with a method to identify 
and locate such helipads from the air.

The jupyter notebook `helipad_detection_pipeline.ipynb` shows how to execute every step of the method 
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

Apply Image augmentation on the images using Google's policy or custom made policy with ImgAug.
