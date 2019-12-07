#!/usr/bin/env python
'''
Just the cross-correlation module as a Python class.
Only depends on numpy, which means porting to C++/CUDA is much easier
than if we were to also depend on OpenCV.

Running this script in isolation will do a simple sanity check.
'''


import numpy as np

class Cross_Correlator:

    def __init__(self, frame, template_shape):
        self.template_shape = template_shape
        self.frame_shape = frame.shape
        self._precomputation(frame)
    
    def _precomputation(self, frame):
        template_shape = self.template_shape
            
        ### ==== STEPS THAT DEPEND ON FRAME
        
        # Calculate the variance at each spot. (This can ordinarily be pre-computed)
        sums = np.cumsum(np.cumsum(frame, axis=0), axis=1)
        sums_sq = np.cumsum(np.cumsum(np.square(frame), axis=0), axis=1)
        
        # Add an extra column and row of zeros:
        sums = np.pad(sums, ((1, 0), (1, 0)), mode='constant')
        sums_sq = np.pad(sums_sq, ((1, 0), (1, 0)), mode='constant')
        
        ### ==== STEPS THAT DEPEND ON KNOWN TEMPLATE SIZE
        
        def compute_variance():
            n = template_shape[0] * template_shape[1]
            
            tx = template_shape[0]
            ty = template_shape[1]
            
            sum_sq_a = sums_sq[:-tx, :-ty]
            sum_sq_b = sums_sq[tx:, :-ty]
            sum_sq_c = sums_sq[:-tx, ty:]
            sum_sq_d = sums_sq[tx:, ty:]
            
            sum_a = sums[:-tx, :-ty]
            sum_b = sums[tx:, :-ty]
            sum_c = sums[:-tx, ty:]
            sum_d = sums[tx:, ty:]
            
            windowed_sum_sq = sum_sq_d + sum_sq_a - sum_sq_b - sum_sq_c
            windowed_sum = sum_d + sum_a - sum_b - sum_c
            
            windowed_avg_sq = windowed_sum_sq / n
            windowed_avg = windowed_sum / n
            
            # nE[(X - E[X])^2] = n(E[X^2] - E[X]^2)
            return n * (windowed_avg_sq - np.square(windowed_avg))
            
        # Variances. Also pre-computable
        self.frame_sigma = np.sqrt(compute_variance())
        
        frame_padded = np.pad(frame,
            ((0, template_shape[0] - 1), (0, template_shape[1] - 1)), mode='mean')
            
        self.frame_fft = np.fft.fft2(frame_padded)
    
    def match(self, template):
        '''
        Perform a match with the given template, returning the
        normalized cross correlation map. (Sometimes called "gamma").
        The brightest spot corresponds to the best-matching location of
        the given template from the frame
        '''
        
        ### ==== STEPS THAT DEPEND ON TEMPLATE
        template_shape = self.template_shape
        frame_shape = self.frame_shape
        assert(np.all(template.shape == template_shape))
        template = template - np.mean(template)
        template_sigma = np.sqrt(np.sum(np.square(template - np.mean(template))))
        template_flipped_padded = np.pad(template[::-1,::-1], 
            ((0, frame_shape[0] - 1), (0, frame_shape[1] - 1)), mode='constant')
        template_fft = np.fft.fft2(template_flipped_padded)
        
        result = np.fft.ifft2(self.frame_fft * template_fft)
        result = np.real(result)
        result = result[template_shape[0] - 1:-(template_shape[0] - 1), template_shape[1] - 1:-(template_shape[1] - 1)]
        result /= self.frame_sigma
        result /= template_sigma
        
        return result


# Demo when running the file standalone.
# Will compare results between OpenCV and our implementation.
# Checks that the results are identical within rounding error (OpenCV uses lower bit precision)
def demo():
    import cv2
    

    def remap_to_zero_to_one(arr):
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    
    def match_opencv(frame, template):
        frame = (frame * 255).astype(np.uint8)
        template = (template * 255).astype(np.uint8)
        cross = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        cross = cross.astype(np.float64)
        return cross
    def save_image(image, fname):
        cv2.imwrite(fname, (np.clip(image, 0, 1) * 255).astype(np.uint8))
        
    def load_image(fname):
        data = cv2.imread(fname)
        data = np.mean(data, axis=2)
        data = data.astype(np.float64)
        data /= 255
        return data
    
    frame = load_image('test_data/natural_image.jpg')
    template = frame[300:350, 300:350].copy()
    
    # using cross correlator
    my_cc = Cross_Correlator(frame, template.shape)
    
    my_gamma = my_cc.match(template)
    
    opencv_gamma = match_opencv(frame, template)
    
    save_image(my_gamma, 'example_fft_based.png')
    save_image(opencv_gamma, 'example_opencv_based.png')
    
    abs_diff = np.abs(my_gamma - opencv_gamma)
    save_image(abs_diff, 'example_abs_diff.png')
    
    print('Max absolute error: {}'.format(np.max(abs_diff)))
    print('Mean absolute error: {}'.format(np.mean(abs_diff)))
    save_image(remap_to_zero_to_one(abs_diff), 'example_abs_diff_normalized.png')

if __name__ == '__main__':
    demo()
