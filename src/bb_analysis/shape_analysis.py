import json
import os
import cv2

# from torchvision import transforms, utils, datasets
# from torch.utils.data import Dataset, DataLoader
#
#
# # to train : fix the shape problem, need to resize the image to what?
#
# class ImageFolderWithPaths(datasets.ImageFolder):
#     """Custom dataset that includes image file paths. Extends
#     torchvision.datasets.ImageFolder
#     """
#
#     # override the __getitem__ method. this is the method that dataloader calls
#     def __getitem__(self, index):
#         # this is what ImageFolder normally returns
#         original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
#         # the image file path
#         path = self.imgs[index][0]
#         # make a new tuple that includes original and the path
#         tuple_with_path = (original_tuple + (path,))
#         return tuple_with_path
#
#
# class ShapeAnalysis:
#
#     def __init__(self, image_folder):
#
#         self.image_folder = image_folder
#
#         self.dataset = ImageFolderWithPaths(self.image_folder,
#                                             transforms.Compose([transforms.ToTensor()]))
#
#         self.data_loader = DataLoader(self.dataset,
#                                       batch_size=100,
#                                       num_workers=0)
#
#     def run(self):
#
#         min_x = (1000, 1000)
#         min_y = (1000, 1000)
#
#         for x_batch, y_batch, path_batch in self.data_loader:
#
#             for x in x_batch:
#                 shape = x.shape
#                 if shape[0] < min_x[0]:
#                     min_x = shape
#                 elif shape[1] < min_y[1]:
#                     min_y = shape
#
#         return min_x, min_y


class ShapeAnalysis:

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
