#!/usr/bin/env python
import numpy as np
import cross_correlation as cc

class Sensor:
    def sense(self, strip_data):
        '''
        Senses a single strip, returning the location in the reference,
        and an uncertainty matrix.
        '''
        raise NotImplementedError()

class Basic_Strip_Sensor(Sensor):
    def __init__(self, reference_frame, strip_shape):
        # Pad reference
        reference_frame_pad = np.pad(reference_frame,
            ((strip_shape[0] - 1, strip_shape[0] - 1), (strip_shape[1] - 1, strip_shape[1] - 1)),
            mode='mean')
        # Make cross correlator
        self.strip_shape = np.array(strip_shape)
        self.cross_correlator = cc.Cross_Correlator(reference_frame_pad, strip_shape)
        
    def sense_with_peak(self, strip_data):
        assert(np.all(np.array(strip_data.shape) == self.strip_shape))
        
        # Just take the peak at face value
        gamma = self.cross_correlator.match(strip_data)
        peak_loc = np.unravel_index(np.argmax(gamma), gamma.shape)
        peak_size = np.max(gamma)
        peak_loc -= np.array(self.strip_shape) - np.array((1, 1))
        
        return peak_loc, peak_size
        
    def sense(self, strip_data):
        peak_loc, peak_size = self.sense_with_peak(strip_data)
        
        extra = {
            "peak_size" : np.array((peak_size, ))
        }
        
        return peak_loc, extra
        

class Downsample_Strip_Sensor(Sensor):
    def __init__(self, reference_frame, strip_shape):
        #Downsample reference
        reference_frame = reference_frame[::2,::2,]
        # Pad reference by strip size divided by 2
        reference_frame_pad = np.pad(reference_frame,
            ((strip_shape[0]//2 - 1, strip_shape[0]//2 - 1), (strip_shape[1]//2 - 1, strip_shape[1]//2 - 1)),
            mode='mean')
        # Make cross correlator
        self.strip_shape = np.array(strip_shape) // 2
        self.cross_correlator = cc.Cross_Correlator(reference_frame_pad, self.strip_shape)
        
    def sense_with_peak(self, strip_data):
        strip_data = strip_data[::2, ::2]
        assert(np.all(np.array(strip_data.shape) == self.strip_shape))
        
        # Just take the peak at face value
        
        gamma = self.cross_correlator.match(strip_data)
        peak_loc = np.unravel_index(np.argmax(gamma), gamma.shape)
        peak_size = np.max(gamma)
        peak_loc -= np.array(self.strip_shape) - np.array((1, 1))

        #restoring loc by mulplication factor 2
        peak_loc *= 2
        
        return peak_loc, peak_size
        
    def sense(self, strip_data):
        peak_loc, peak_size = self.sense_with_peak(strip_data)
        
        extra = {
            "peak_size" : np.array((peak_size, ))
        }
        
        return peak_loc, extra

class Double_Half_Strip(Sensor):
    def __init__(self, reference_frame, strip_shape, num_partitions=2):
        self.num_partitions = num_partitions
        self.strip_shape = np.array(strip_shape)
        
        self.half_strip_shape = self.strip_shape.copy()
        self.half_strip_shape[1] //= self.num_partitions
        
        self.padding = self.half_strip_shape - np.array((1, 1))
        # Pad reference
        reference_frame_pad = np.pad(reference_frame,
            ((self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])),
            mode='mean')
        self.cross_correlator = cc.Cross_Correlator(reference_frame_pad, (self.half_strip_shape[0], self.half_strip_shape[1]))
        
    
    def sense_with_metadata(self, strip_data):
        assert(np.all(np.array(strip_data.shape) == self.strip_shape))
        
        # Split into halves
        half_strips = []
        offsets = []
        for i in range(self.num_partitions):
            left_bound = i * self.half_strip_shape[1]
            right_bound = (i+1) * self.half_strip_shape[1]
            offsets.append(np.array((0, left_bound)))
            half_strips.append(strip_data[:, left_bound:right_bound])
        
        # Match both halves
        gammas = []
        for half_strip in half_strips:
            gammas.append(self.cross_correlator.match(half_strip))
            
        # Get the peaks for each
        peaks = []
        peak_sizes = []
        for gamma, offset in zip(gammas, offsets):
            peak_loc = np.unravel_index(np.argmax(gamma), gamma.shape)
            peak_size = gamma[peak_loc[0], peak_loc[1]]
            peak_loc -= self.padding
            peak_loc -= offset
            peaks.append(peak_loc)
            peak_sizes.append(peak_size)
        
        # Compute MLE
        peaks = np.array(peaks)
        peak_sizes = np.array(peak_sizes)
        mean_loc = np.mean(peaks, axis=0)
        var_loc = np.var(peaks, axis=0)
        covar_matr = np.diag(var_loc)
        
        return mean_loc, covar_matr, peaks, peak_sizes
        
    def sense(self, strip_data):
        mean_loc, covar_matr, peaks, peak_sizes = self.sense_with_metadata(strip_data)
        
        
        extra = {
            "uncertainty" : covar_matr
        }
        
        
        return mean_loc, extra
    
        
# Simple test, cutting up the given natural image into strips
# and then doing a match
def demo():
    import cv2
    
        
    def load_image(fname):
        data = cv2.imread(fname)
        data = np.mean(data, axis=2)
        data = data.astype(np.float64)
        data /= 255
        return data
    
    frame = load_image('test_data/natural_image.jpg')
    
    strip_width = 512
    strip_height = 16
    
    sensors = [
        Basic_Strip_Sensor(frame, (strip_height, strip_width)),
        Double_Half_Strip(frame, (strip_height, strip_width)),
        Downsample_Strip_Sensor(frame, (strip_height, strip_width)),
    ]
    for sensor in sensors:
        print('Testing: ', sensor)
        for strip_y in range(0, strip_width, strip_height):
            
            strip_data = frame[strip_y:strip_y+strip_height, :]
            
            result = sensor.sense(strip_data)
            
            if result is None:
                print('Error! Sensor failed to find a perfect match!')
            else:
                match_loc, uncertainty = result
                if match_loc[0] != strip_y:
                    print('Error! Sensor failed to find correct position!')
    print('End of test.')
    
if __name__ == '__main__':
    demo()
