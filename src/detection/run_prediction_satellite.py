import os
import cv2
import json
from numpy import expand_dims
from tqdm import tqdm

from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image

import sys
sys.path.append('../')

from utils.globalmaptiles import GlobalMercator

from training.helipad_config import HelipadConfig
from training.filter_manager import FilterManager

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class RunPredictionSatellite:

    def __init__(self, cache_tms_sat_folder, output_meta_folder, zoom_level,
                 model_folder, weights_filename, model_number, activate_filters=False,
                 redo_prediction=False):
        self.cache_tms_sat_folder = cache_tms_sat_folder
        self.output_meta_folder = output_meta_folder
        self.zoom_level = zoom_level
        self.model_folder = model_folder
        self.weights_filename = weights_filename
        self.model_number = model_number
        self.activate_filters = activate_filters
        self.redo_prediction = redo_prediction
        self.image_folder = os.path.join(self.cache_tms_sat_folder, str(zoom_level))
        self.meta_folder = os.path.join(self.output_meta_folder, str(zoom_level))
        if not os.path.isdir(os.path.join(self.output_meta_folder, str(zoom_level))):
            os.mkdir(os.path.join(self.output_meta_folder, str(zoom_level)))
        self.target_files = self.build_target_files_path()
        self.config = HelipadConfig()
        self.globalmercator = GlobalMercator()
        self.model_predict_setup()

    def build_target_files_path(self):
        target_files = []
        for subdir, dirs, files in os.walk(self.image_folder, topdown=True):
            for file in files:
                xtile = os.path.basename(subdir)
                ytile = os.path.splitext(file)[0]
                filepath = os.path.join(subdir, file)
                meta_filepath = os.path.join(self.output_meta_folder,
                                             str(self.zoom_level),
                                             str(xtile),
                                             'Satellite_{}_{}_{}.meta'.format(str(self.zoom_level), str(xtile), str(ytile)))
                target_files.append([filepath, meta_filepath])
        return target_files

    def model_predict_setup(self):
        self.model_predict = MaskRCNN(mode='inference', model_dir=self.model_folder, config=self.config)
        self.model_predict.load_weights(os.path.join(self.model_folder, self.weights_filename),
                                        by_name=True)

    def get_coordinates_info(self, xtile, ytile, zoom_level):
        coordinates_info = dict()
        coordinates_info["zoom"] = zoom_level
        coordinates_info["xtile"] = xtile
        coordinates_info["ytile"] = ytile

        tile_bounds_coordinates = self.globalmercator.TileLatLonBox(xtile, ytile, zoom_level)
        tile_center_lat = (tile_bounds_coordinates[0][0] + tile_bounds_coordinates[3][0]) / 2
        tile_center_lon = (tile_bounds_coordinates[0][1] + tile_bounds_coordinates[3][1]) / 2
        coordinates_info["latitude"] = tile_center_lat
        coordinates_info["longitude"] = tile_center_lon
        coordinates_info["bounds"] = tile_bounds_coordinates

        return coordinates_info

    def predict_image(self, image):
        scaled_image = mold_image(image, self.config)
        sample = expand_dims(scaled_image, 0)
        yhat = self.model_predict.detect(sample, verbose=0)

        rois = yhat[0]['rois']
        class_id = yhat[0]['class_ids']
        score = yhat[0]['scores']

        # reorder rois :
        # x1, y1, x2, y2
        bboxes = []
        for roi in rois:
            box = [int(roi[1]), int(roi[0]), int(roi[3]), int(roi[2])]
            bboxes.append(box)

        class_ids = []
        for id in class_id:
            class_ids.append(int(id))

        scores = []
        for s in score:
            scores.append(float(s))

        # filter is helipad detected
        if self.activate_filters and len(scores) > 0:
            # remove if score < threshold (see FilterManager for default values)
            bboxes, class_ids, scores = FilterManager.filter_by_scores(bboxes, class_ids, scores)
            # Filter overlapping box (see FilterManager for default value of threshold_iou and threshold_area)
            bboxes, class_ids, scores = FilterManager.filter_by_iou(bboxes, class_ids, scores)

        # Save to meta roi
        prediction = dict()
        prediction["box"] = bboxes
        prediction["class_id"] = class_ids
        prediction["score"] = scores

        if len(bboxes) > 0:
            prediction["helipad"] = True
            print(scores)
        else:
            prediction["helipad"] = False

        return prediction

    def convert_point_to_coordinate(self, x, y, minLat, minLon, maxLat, maxLon, image_shape, verbose=0):
        tmp = x
        x = y
        y = tmp
        lat_factor = (image_shape[0] - x)/image_shape[0]
        lon_factor = y / image_shape[1]
        lat = minLat + (abs(minLat - maxLat) * lat_factor)
        lon = minLon + (abs(minLon - maxLon) * lon_factor)

        if verbose == 1:
            # print(x)
            # print(y)
            # print(minLat)
            # print(minLon)
            # print(maxLat)
            # print(maxLon)
            # print(image_shape)
            # print(lat_factor)
            # print(lon_factor)
            print(lat)
            print(lon)

        return lat, lon

    def convert_bboxes_to_coordinates(self, bboxes, bounds_coordinates, image_shape):
        # convert box to coordinates

        minLat, minLon = bounds_coordinates[0][0], bounds_coordinates[0][1]
        maxLat, maxLon = bounds_coordinates[3][0], bounds_coordinates[3][1]

        # lat lon of center
        center_pixel = []
        for box in bboxes:
            xmean = (box[0] + box[2]) / 2
            ymean = (box[1] + box[3]) / 2
            center_pixel.append((xmean, ymean))

        bboxes_center_coordinates = []
        for center in center_pixel:
            xmean, ymean = center[0], center[1]
            bboxes_center_lat, bboxes_center_lon = self.convert_point_to_coordinate(xmean, ymean,
                                                                                    minLat, minLon,
                                                                                    maxLat, maxLon,
                                                                                    image_shape,
                                                                                    verbose=1)
            bboxes_center_coordinates.append((bboxes_center_lat, bboxes_center_lon))

        bboxes_bounds_coordinates = []
        for box in bboxes:
            corners = [(box[0], box[1]),  # minx, miny
                       (box[0], box[3]),  # minx, maxy
                       (box[2], box[1]),  # maxx, miny
                       (box[2], box[3])]  # maxx, maxy
            corners_coordinates = []
            for point in corners:
                x, y = point[0], point[1]
                lat, lon = self.convert_point_to_coordinate(x, y, minLat, minLon, maxLat, maxLon, image_shape)
                point_coordinates = (lat, lon)
                corners_coordinates.append(point_coordinates)
            bboxes_bounds_coordinates.append(corners_coordinates)

        return bboxes_center_coordinates, bboxes_bounds_coordinates
    
    def initiate_meta(self, meta_filename):
        meta = dict()
        meta_coordinates = meta_filename.split('_')
        zoom_level = int(meta_coordinates[1])
        xtile = int(meta_coordinates[2])
        ytile = int(meta_coordinates[3])
        coordinates_info = self.get_coordinates_info(xtile, ytile, zoom_level)
        meta["coordinates"] = coordinates_info
        return meta

    
    def run(self):

        for i in tqdm(range(len(self.target_files))):
            target_file = self.target_files[i]

            image_path = target_file[0]
            meta_path = target_file[1]
            meta_filename = os.path.splitext(os.path.basename(meta_path))[0]
            image = cv2.imread(image_path)
            image_shape = image.shape

            if os.path.isfile(meta_path):
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    coordinates_info = meta["coordinates"]
                    xtile = coordinates_info["xtile"]
                except:
                    meta = self.initiate_meta(meta_filename)
            else:
                meta = self.initiate_meta(meta_filename)
            
            if "predicted" in meta:
                predicted = meta["predicted"]
            else:
                predicted = {}

            key = "model_{}".format(self.model_number)
            
            if key in predicted and not self.redo_prediction:
                continue
            
            # now predict on image
            prediction = self.predict_image(image)

            bounds_coordinate = coordinates_info["bounds"]
            bboxes = prediction["box"]

            bboxes_center_coordinates, bboxes_bounds_coordinates = self.convert_bboxes_to_coordinates(bboxes,
                                                                                                      bounds_coordinate,
                                                                                                      image_shape)

            prediction_coordinates = dict()
            prediction_coordinates["center"] = bboxes_center_coordinates
            prediction_coordinates["bounds"] = bboxes_bounds_coordinates
            prediction["coordinates"] = prediction_coordinates

            predicted[key] = prediction

            meta["predicted"] = predicted
            
            if not os.path.isdir(os.path.join(self.output_meta_folder,
                                              str(self.zoom_level))):
                os.mkdir(os.path.join(self.output_meta_folder,
                                      str(self.zoom_level)))
            if not os.path.isdir(os.path.join(self.output_meta_folder,
                                              str(self.zoom_level),
                                              str(xtile))):
                os.mkdir(os.path.join(self.output_meta_folder,
                                      str(self.zoom_level),
                                      str(xtile)))

            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=4, sort_keys=True)


if __name__ == "__main__":

    # cache_tms_sat_folder = "C:\\Users\\jonas\\Desktop\\SAS.Planet.Release.191221\\\cache_tms\\sat"
    # cache_tms_sat_folder = "C:\\Users\\jonas\\Desktop\\cache_tms_test"
    # output_meta_folder = "C:\\Users\\jonas\\Desktop\\cache_tms_meta"
    # model_folder = "C:\\Users\\jonas\\Desktop\\Helipad\\model"

    cache_tms_sat_folder = "../../../Detection/Detection_Dataset"
    output_meta_folder = "../../../Detection/Detection_Dataset_meta"
    model_folder = "../../model"

    # weights_filename = "helipad_cfg_6_aug4_3+20200103T1225/mask_rcnn_helipad_cfg_6_aug4_3+_0288.h5"
    # model_number = 5

    # weights_filename = "helipad_cfg_8_no47_aug2_3+20200108T0600/mask_rcnn_helipad_cfg_8_no47_aug2_3+_0472.h5"
    # model_number = 7

    # weights_filename = "helipad_cfg_9_no47_aug2_3+20200112T2326/mask_rcnn_helipad_cfg_9_no47_aug2_3+_0257.h5"
    # model_number = 8

    weights_filename = "helipad_cfg_aug2_5+20191211T1749/mask_rcnn_helipad_cfg_aug2_5+_0381.h5"
    model_number = 4

    zoom_level = [18, 19]
    activate_filters = False

    for zoom in zoom_level:

        run_prediction_satellite = RunPredictionSatellite(cache_tms_sat_folder=cache_tms_sat_folder,
                                                          output_meta_folder=output_meta_folder,
                                                          zoom_level=zoom,
                                                          model_folder=model_folder,
                                                          weights_filename=weights_filename,
                                                          model_number=model_number,
                                                          activate_filters=activate_filters)

        run_prediction_satellite.run()

    #TODO:
    # load the image
    # write image coordinates (zoom level, tsm coordinates, center coordinates and corner coordinates)
    # load the model
    # predict
    # save prediction in meta
    # get the center of box
    # convert center to coordinates
    # save center coordinates
    # OPTIONAL : include the google maps url of the helipad
    # OPTIONAL : include info of location (country, city, id? ...)


