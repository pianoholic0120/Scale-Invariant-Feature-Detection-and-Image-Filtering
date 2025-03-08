import numpy as np
import cv2
from numpy.lib.stride_tricks import as_strided

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1
        
    def get_keypoints(self, image):
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # First octave starts with the original image
        first_octave = [image.copy()]
        # Apply gaussian blur with increasing sigma values
        for i in range(1, self.num_guassian_images_per_octave):
            sigma = self.sigma ** i
            g_img = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
            first_octave.append(g_img)
        
        # Second octave - downsample the last image from first octave
        h, w = first_octave[-1].shape[:2]
        downsampled = cv2.resize(first_octave[-1], (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
        
        # Second octave starts with the downsampled image
        second_octave = [downsampled]
        # Apply gaussian blur with increasing sigma values
        for i in range(1, self.num_guassian_images_per_octave):
            sigma = self.sigma ** i
            g_img = cv2.GaussianBlur(downsampled, (0, 0), sigmaX=sigma)
            second_octave.append(g_img)
            
        gaussian_pyramids = [first_octave, second_octave]
        
        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        dog_pyramids = []
        for octave in range(self.num_octaves):
            dog_images = []
            for i in range(self.num_DoG_images_per_octave):
                dog = cv2.subtract(gaussian_pyramids[octave][i + 1], gaussian_pyramids[octave][i])
                dog_images.append(dog)
                # M, m = max(dog.flatten()), min(dog.flatten())
                # norm = (dog-m)*255/(M-m)
                # cv2.imwrite(f'./testdata/DoG{octave+1}-{i+1}.png', norm)
            dog_pyramids.append(dog_images)
            
        # Step 3: Thresholding the value and Find local extremum (local maximum and local minimum)
        keypoints = np.array([]).reshape((0, 2))
        for octave in range(self.num_octaves):
            dogs = np.array(dog_pyramids[octave])            
            neighborhood_shifts = [(x, y, z) for x in range(-1, 2) for y in range(-1, 2) for z in range(-1, 2)]
            cube = np.array([np.roll(dogs, shift, axis=(0, 1, 2)) for shift in neighborhood_shifts])
            mask = (np.absolute(dogs) >= self.threshold) & ((np.min(cube, axis=0) == dogs) | (np.max(cube, axis=0) == dogs))
            for i in range(1, self.num_DoG_images_per_octave - 1):
                current_mask = mask[i]
                y_coords, x_coords = np.meshgrid(np.arange(current_mask.shape[1]), np.arange(current_mask.shape[0]))
                matched_points = np.stack([x_coords[current_mask], y_coords[current_mask]], axis=-1)
                if octave > 0:
                    matched_points *= (2 ** octave)
                keypoints = np.concatenate([keypoints, matched_points])

        # Step 4: Delete duplicate keypoints
        keypoints = np.unique(np.array(keypoints), axis = 0)
        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1], keypoints[:,0]))]
            
        return keypoints