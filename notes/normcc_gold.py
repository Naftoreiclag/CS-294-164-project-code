#!/usr/bin/env python
'''
Normalized cross correlation gold standard.

Goal is to establish an easily verifable standard implementation of the
normalized cross-correlation algorithm.
'''

import numpy as np
import cv2

######## MATCHING TECHNIQUES

def match_gold(frame, template):
    '''
    Do a match. No padding or any other extra steps.
    Should be an exact naive implementation of
        Lewis, J.P.. (1995). Fast Normalized Cross-Correlation. Ind. Light Magic. 10. 
        <http://scribblethink.org/Work/nvisionInterface/nip.html>
    Not optimized for speed, but for clarity of code.
    Returns the gamma function described in equation (2).
    '''
    assert(frame.dtype == np.float64)
    assert(template.dtype == np.float64)
    
    # Compare with opencv. Obviously not a normal part of the algorithm
    opencv_result = match_opencv(frame, template)['gamma']
    
    ### ==== STEPS THAT DEPEND ON FRAME
    
    # Calculate the variance at each spot. (This can ordinarily be pre-computed)
    sums = np.cumsum(np.cumsum(frame, axis=0), axis=1)
    sums_sq = np.cumsum(np.cumsum(np.square(frame), axis=0), axis=1)
    
    # Add an extra column and row of zeros:
    sums = np.pad(sums, ((1, 0), (1, 0)), mode='constant')
    sums_sq = np.pad(sums_sq, ((1, 0), (1, 0)), mode='constant')
    
    ### ==== STEPS THAT DEPEND ON KNOWN TEMPLATE SIZE
    
    def compute_variance():
        n = template.shape[0] * template.shape[1]
        
        tx = template.shape[0]
        ty = template.shape[1]
        
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
    frame_sigma = np.sqrt(compute_variance())
    
    frame_padded = np.pad(frame,
        ((0, template.shape[0] - 1), (0, template.shape[1] - 1)), mode='mean')
        
    frame_fft = np.fft.fft2(frame_padded)
    
    ### ==== STEPS THAT DEPEND ON TEMPLATE
    template -= np.mean(template)
    template_sigma = np.sqrt(np.sum(np.square(template - np.mean(template))))
    template_flipped_padded = np.pad(template[::-1,::-1], 
        ((0, frame.shape[0] - 1), (0, frame.shape[1] - 1)), mode='constant')
    template_fft = np.fft.fft2(template_flipped_padded)
    
    result = np.fft.ifft2(frame_fft * template_fft)
    result = np.real(result)
    result = result[template.shape[0] - 1:-(template.shape[0] - 1), template.shape[1] - 1:-(template.shape[1] - 1)]
    result /= frame_sigma
    result /= template_sigma
    
    # Return
    extra = np.abs(result - opencv_result)
    
    print('Average absolute difference from OpenCV result: ', np.mean(extra))
    
    return {
        "gamma" : result,
        "frame_sigma" : remap_to_zero_to_one(frame_sigma),
        "sums" : remap_to_zero_to_one(sums),
        "sums_sq" : remap_to_zero_to_one(sums),
        "opencv_comparison" : remap_to_zero_to_one(extra)
    }
    
def match_opencv(frame, template):
    assert(frame.dtype == np.float64)
    assert(template.dtype == np.float64)
    
    frame = (frame * 255).astype(np.uint8)
    template = (template * 255).astype(np.uint8)
    
    cross = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    
    cross = cross.astype(np.float64)
    
    assert(cross.dtype == np.float64)
    return {
        "gamma" : cross
    }
    
######## TEST DATA LOADERS

def test_random():
    np.random.seed(54235657)
    frame = np.random.random_sample((500, 500)).astype(np.float64)
    template = frame[100:150, 100:150].copy()
    
    return frame, template
    
def test_natural_image():
    np.random.seed(12345325)
    frame = load_image('test_data/natural_image.jpg')
    template = frame[200:250, 200:250].copy()
    
    # Add some random noise
    template += np.random.random_sample((50, 50)) * 0.2
    template = np.clip(template, 0, 1)
    
    return frame, template
    
def test_natural_image2():
    np.random.seed(12345325)
    frame = load_image('test_data/natural_image.jpg')
    template = frame[300:350, 300:350].copy()
    
    # Add some random noise
    template += np.random.random_sample((50, 50)) * 0.2
    template = np.clip(template, 0, 1)
    
    return frame, template
    
    
######## HELPER

def remap_to_zero_to_one(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def save_image(image, fname):
    assert(image.dtype == np.float64)
    
    cv2.imwrite(fname, (np.clip(image, 0, 1) * 255).astype(np.uint8))
    
def load_image(fname):
    data = cv2.imread(fname)
    
    assert(data.dtype == np.uint8)
    
    data = np.mean(data, axis=2)
    
    data = data.astype(np.float64)
    data /= 255
    
    assert(data.dtype == np.float64)
    return data
    
######## MAIN FUNC

def demo():
    # Add your techniques here
    techniques = ['match_opencv', 'match_gold']
    
    # Add your tests here
    tests = ['test_random', 'test_natural_image', 'test_natural_image2']
    
    # For every test
    for test_name in tests:
        test_func = eval(test_name)
        frame, template = test_func()
        
        print('=== Performing test on: ', test_name)
        
        # Debug save
        save_image(frame, '{}_frame.png'.format(test_name))
        save_image(template, '{}_template.png'.format(test_name))
    
        # Do cross-correlation using each technique
        for technique_name in techniques:
            technique_func = eval(technique_name)
            
            print('--- Running: ', technique_name)
            retvals = technique_func(frame.copy(), template.copy())
            
            if retvals is None:
                print('TEST FAILED!')
            else:
                gamma = retvals['gamma']
                # Debug print/save
                print('Max val: ', np.max(gamma))
                print('Min val: ', np.min(gamma))
                save_image(gamma, '{}-{}.png'.format(test_name, technique_name))
                
                for extra_output in retvals:
                    if extra_output == 'gamma':
                        continue
                    extra_img = retvals[extra_output]
                    print('Extra output: ', extra_output)
                    save_image(extra_img, '{}-{}-{}.png'.format(test_name, technique_name, extra_output))
    
if __name__ == '__main__':
    demo()
