import os
import matplotlib.pyplot as plt
import numpy as np
import cv2


class ShadowDetection:
    
    """
    Shadow Detection on an image. The image is supposed to be a bounding box detected by a model. 
    """
    
    def __init__(self, image, minimum_size_window=3, 
                 threshold_v=0.35, threshold_s=0.02, ratio=1, d_0=3):
        """
        `image`: an image cropped around the bounding box\n
        `minimum_size_window`: the minimum size of a shadow is defined by a square of side `(minimum_size_window*2-1)`\n
        `threshold_v`: the mean of the window in V must be inferior than `threshold_v` to be accepted.\n
        `threshold_s`: the mean of the window in S must be superior than `threshold_s` to be accepted.\n
        `ratio`: all the values of the window in `c3` must be superior than `mean(c3)*ratio`.\n
        `d_0`: the candidate pixel to be added to the region shadow must be below a Mahalanobis distance `d_0` from the `mean(c3[region])`.
        """
        self.image = image
        self.minimum_size_window = minimum_size_window
        self.threshold_v = threshold_v
        self.threshold_s = threshold_s
        self.ratio = ratio
        self.d_0 = d_0
        
        self.c3, self.S, self.V, self.edges = self.preprocessing(self.image)
    
    @staticmethod
    def compute_c3(image):
        """
        Compute the c3 channel of the image
        """
        c3 = np.arctan(image[:,:,0]/np.maximum(image[:,:,1], image[:,:,2]))
        return c3

    @staticmethod
    def apply_average_kernel(img, kernel_size):
        """
        Apply an average kernel on the image
        """
        return cv2.blur(img,(kernel_size, kernel_size))
    
    @staticmethod
    def get_hsv(image):
        """
        Get the image in the HSV colorspace
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    @staticmethod
    def compute_sobelx(v, kernel_size):
        """
        Compute the gradient of the image using the SobelX algorithm
        """
        return cv2.Sobel(v,cv2.CV_64F,1,0,ksize=kernel_size)

    @staticmethod
    def preprocessing(image):
        """
        Preprocessing of the image\n
        Returns the channels `c3`, `S`, `V` and `edges`
        """
        c3 = ShadowDetection.compute_c3(image)
        c3_smoothed = ShadowDetection.apply_average_kernel(c3, 3)
        hsv = ShadowDetection.get_hsv(image)
        edges = ShadowDetection.compute_sobelx(hsv[:,:,2], kernel_size=3)
        return c3, hsv[:,:,1], hsv[:,:,2], edges
   
    @staticmethod
    def zero_pad(matrix, pad_left, pad_right, pad_top, pad_bottom):
        """
        Apply zero padding on a matrix
        """
        h, w = matrix.shape
        H, W = h+pad_top+pad_bottom, w+pad_left+pad_right
        matrix_pad = np.zeros((H, W))
        matrix_pad[pad_bottom:H-pad_top,pad_left:W-pad_right] = matrix
        return matrix_pad
    
    @staticmethod
    def check_local_maximum(candidate_seed, candidate_window):
        if candidate_seed >= np.max(candidate_window):
            return True
        else:
            return False
    
    @staticmethod
    def check_value_neighbourhood(candidate_window, mean_c3, ratio=1/2):
        if np.min(candidate_window) > mean_c3*ratio:  # should be > mean_c3 according to the article but change to fit better
            return True
        else:
            return False
    
    @staticmethod
    def check_mean_V(V_window, threshold_v):
        if np.mean(V_window/255.0) < threshold_v:
            return True
        else:
            return False
    
    @staticmethod
    def check_mean_S(S_window, threshold_s):
        if np.mean(S_window/255.0) > threshold_s:
            return True
        else:
            return False
    
    @staticmethod
    def check_window_already_seed(i, j, seeds, delay, minimum_size_window):
        d = delay
        m = minimum_size_window
        if np.max(seeds[i-d:i+m, j-d:j+m]) > 0:
            return False
        else:
            return True
        
    def seed_selection(self):
        h, w = self.c3.shape
        seeds = np.zeros((h, w))
        prototype = {}
        k = 1
        d = self.minimum_size_window-1
        m = self.minimum_size_window
        c3_pad = self.zero_pad(self.c3, d, d, d, d)
        S_pad = self.zero_pad(self.S, d, d, d, d)
        V_pad = self.zero_pad(self.V, d, d, d, d)
        mean_c3 = np.mean(self.c3)

        for i in range(h):
            for j in range(w):
                candidate_seed = self.c3[i][j]
                candidate_window_c3 = c3_pad[i+d-d:i+d+m, j+d-d:j+d+m]
                candidate_window_S = S_pad[i+d-d:i+d+m, j+d-d:j+d+m]
                candidate_window_V = V_pad[i+d-d:i+d+m, j+d-d:j+d+m]
                # condition a:
                if not self.check_local_maximum(candidate_seed, candidate_window_c3):
                    continue
                # condition b:
                elif not self.check_value_neighbourhood(candidate_window_c3, mean_c3, ratio=self.ratio):
                    continue
                # condition c:
                elif not self.check_mean_V(candidate_window_V, threshold_v=self.threshold_v): # hyper-parameter
                    continue
                # condition d:
                elif not self.check_mean_S(candidate_window_S, threshold_s=self.threshold_s): # hyper-parameter
                    continue
                # continue e
                elif not self.check_window_already_seed(i, j, seeds, d, m):
                    continue
                else:
                    # accept candidate and candidate window
                    seeds[i-d:i+m, j-d:j+m] = k*np.ones((self.minimum_size_window*2-1, self.minimum_size_window*2-1))
                    indices = []
                    for u in range(i-d, i+m):
                        for v in range(j-d, j+m):
                            indices.append((u,v))
                    prototype[k] = {'indices':indices,
                                    'values':candidate_window_c3.ravel(),
                                    'mean': np.mean(candidate_window_c3),
                                    'sigma': np.std(candidate_window_c3)}
                    k += 1
        return seeds.astype(int), prototype
    
    @staticmethod
    def check_pixel_in_shadow(prototype, i, j):
        for key, values in prototype.items():
            if (i,j) in values['indices']:
                return True
        return False

    @staticmethod
    def check_pixel_neighbours_boundary_shadow(prototype, i, j):
        neighbours = [(i-1,j), (i+1,j), (i-1,j-1), (i, j-1), (i,j+1), (i+1, j+1), (i+1, j-1), (i-1, j+1)]
        for key, values in prototype.items():
            indices = values['indices']
            for neighbour in neighbours:
                if neighbour in indices:
                    return key
        return 0
    
    @staticmethod
    def check_mahalanobis_distance(pixel_c3, region_id, prototype, d_0 = 3):
        mean_c3 = prototype[region_id]['mean']
        sigma_c3 = prototype[region_id]['sigma']
        if np.abs(pixel_c3 - mean_c3)/sigma_c3 < d_0:
            return True
        else:
            return False
        
    def region_growing(self, seeds, prototype):
        n, p, c = self.image.shape

        d = self.minimum_size_window-1
        m = self.minimum_size_window
        c3_pad = self.zero_pad(self.c3, d, d, d, d)
        S_pad = self.zero_pad(self.S, d, d, d, d)
        V_pad = self.zero_pad(self.V, d, d, d, d)

        pixel_added = True
        while pixel_added:
            pixel_added = False
            for i in range(n):
                for j in range(p):
                    # check if pixel already in shadow area
                    if self.check_pixel_in_shadow(prototype, i, j):
                        continue
                    # check if pixel is a neighbours of a boundary pixel
                    region_id = self.check_pixel_neighbours_boundary_shadow(prototype, i, j) 
                    if region_id == 0:
                        continue
                    pixel_c3 = self.c3[i][j]
                    if not self.check_mahalanobis_distance(pixel_c3, region_id, prototype, d_0=self.d_0): # d_0 hyper-parameter
                        continue

                    # The magnitude of the gradient of V < T_e = 0.30 (not a shadow boundary pixel)
                    # ???

                    candidate_window_V = V_pad[i+d-d:i+d+m, j+d-d:j+d+m]    
                    if not self.check_mean_V(candidate_window_V, threshold_v=self.threshold_v): # hyper-parameter
                        continue
                    candidate_window_S = S_pad[i+d-d:i+d+m, j+d-d:j+d+m]
                    # condition d:
                    if not self.check_mean_S(candidate_window_S, threshold_s=self.threshold_s): # hyper-parameter
                        continue

                    # Add the pixel to the region region_id
                    prototype[region_id]['indices'].append((i,j))
                    prototype[region_id]['values']  = np.append(prototype[region_id]['values'], pixel_c3)
                    prototype[region_id]['mean'] = np.mean(prototype[region_id]['values'])
                    prototype[region_id]['sigma'] = np.std(prototype[region_id]['values'])

                    # TODO: Add region merging 

        return prototype
    
    def run(self, verbose=2):
        
        seeds, prototype = self.seed_selection()
        
        prototype = self.region_growing(seeds, prototype)
        
        if verbose == 1:
            image_seed = self.mark_seeds_on_image(self.image, seeds)
            self.display_two_image_side_by_side(self.image, image_seeds)
        elif verbose == 2:
            image_seed = self.mark_seeds_on_image(self.image, seeds)
            image_shadow = self.mark_shadows_on_image(self.image, prototype)
            self.display_three_image_side_by_side(self.image, image_seed, image_shadow)
        
        # TODO: Analyse the shadow to return true or false if it's an helipad or not
    
    @staticmethod
    def mark_seeds_on_image(image, seeds):
        image_seed = image.copy()
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if seeds[i][j] > 1:
                    image_seed[i,j,:] = [0,0,255]
        return image_seed
    
    @staticmethod
    def mark_shadows_on_image(image, prototype):
        image_shadow = image.copy()
        for key,values in prototype.items():
            indices = values['indices']
            for indice in indices:
                i = indice[0]
                j = indice[1]
                image_shadow[i,j,:] = [0,0,255]    
        return image_shadow
    
    @staticmethod
    def display_two_image_side_by_side(image_1, image_2):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
        
        axes[0].axis('off')
        axes[0].imshow(image_1)
        axes[0].set_title('Input image')

        axes[1].axis('off')
        axes[1].imshow(image_2)
        axes[1].set_title('Shadow seeds')

        plt.show()
    
    @staticmethod
    def display_three_image_side_by_side(image_1, image_2, image_3):
        fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharex=True, sharey=True)

        axes[0].axis('off')
        axes[0].imshow(image_1)
        axes[0].set_title('Input image')

        axes[1].axis('off')
        axes[1].imshow(image_2)
        axes[1].set_title('Shadow seeds')

        axes[2].axis('off')
        axes[2].imshow(image_3)
        axes[2].set_title('Shadows')

        plt.show()

        