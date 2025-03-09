import numpy as np
import cv2

class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        # Computation of look-up table for Spatial Kernel Gs
        G_s_table = np.exp(-0.5*(np.arange(self.pad_w+1)**2)/self.sigma_s**2) # wndw_size = 2*pad_w + 1, use the symmetry property of gaussian to reduce the memory usage
        # Computation of look-up table for Range Kernel Gr
        G_r_table = np.exp(-0.5*(np.arange(256)/255)**2/self.sigma_r**2) # normalize the range to [0, 1]
        calculate = np.zeros(padded_img.shape)
        output_no_norm = np.zeros(padded_img.shape)
        for i in range(-self.pad_w, self.pad_w + 1):
            for j in range(-self.pad_w, self.pad_w + 1):
                guidance_r = G_r_table[np.abs(np.roll(padded_guidance, [i, j], axis = [1, 0]) - padded_guidance)] 
                if padded_guidance.ndim == 2:
                    weight_r = guidance_r
                else:
                    weight_r = np.prod(guidance_r, axis = 2) 
                weight_s = G_s_table[abs(i)] * G_s_table[abs(j)] # multiplication of two exponential 
                weight = weight_r * weight_s
                padded_img_shift = np.roll(padded_img, [i, j], axis = [1, 0])
                for channel in range(padded_img.ndim):
                    calculate[:, :, channel] += weight
                    output_no_norm[:, :, channel] += padded_img_shift[:, :, channel] * weight
        
        output = (output_no_norm / calculate)[self.pad_w:-self.pad_w, self.pad_w:-self.pad_w,:]

        return np.clip(output, 0, 255).astype(np.uint8)