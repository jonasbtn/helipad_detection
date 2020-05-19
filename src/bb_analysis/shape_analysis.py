import json
import os
import cv2


class ShapeAnalysis:
    
    """
    Get the minimum width and minimum height of the detected bounding boxes to find the optimal resize.
    """
    
    def __init__(self, image_folder):
        self.image_folder = image_folder

    def run(self):
        min_x = (1000, 1000)
        min_y = (1000, 1000)

        for subdir, dirs, files in os.walk(self.image_folder, topdown=True):
            for file in files:
                image = cv2.imread(os.path.join(subdir, file))
                if image is None:
                    continue
                shape = image.shape
                if shape[0] < min_x[0]:
                    min_x = shape
                elif shape[1] < min_y[1]:
                    min_y = shape

        return min_x, min_y


if __name__ == "__main__":

    image_folder = "C:\\Users\\jonas\\Desktop\\Helipad\\Detected_Boxes\\model_7_0.0"

    shape_analysis = ShapeAnalysis(image_folder)

    min_x, min_y = shape_analysis.run()

    print(min_x)
    print(min_y)
