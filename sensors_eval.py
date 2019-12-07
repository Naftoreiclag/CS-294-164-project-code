#!/usr/bin/env python
# Runs a given cross-correlator from beginning to end on a given video file

import matplotlib.pyplot as plt
import cv2
import numpy as np
import sensors as sensors_py
    
def load_image(fname):
    data = cv2.imread(fname)
    data = np.mean(data, axis=2)
    data = data.astype(np.float64)
    data /= 255
    return data

def substitute_none_with_nan(arr):
    if len(arr) == 0:
        return []
    
    template_shape = None
    for x in arr:
        if x is not None:
            template_shape = x.shape
            
    assert(template_shape is not None)
    
    retval = []
    for x in arr:
        if x is None:
            nan = np.empty(template_shape)
            nan.fill(np.nan)
            retval.append(nan)
        else:
            retval.append(np.copy(x))
    return retval

class Video_Stream:
    
    def __init__(self, opencv_stream):
        self.stream = opencv_stream
        self.stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
    def get_num_frames(self):
        return int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def get_frame(self, frame_idx):
        self.stream.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = self.stream.read()
        assert(success)
        frame = np.mean(frame, axis=2)
        frame = frame.astype(np.float64)
        frame /= 255
        return frame

def main():
    # List of videos and the corresponding reference, number of frames max to use
    videos = [
        ('test1', 'test_data/20160R_V006.avi', 'test_data/20160R_V006_Reference.png', None)
    ]
    
    sensors = [
        #('DHS', sensors_py.Double_Half_Strip),
        ('Downsample', sensors_py.Downsample_Strip_Sensor),
        #('Basic', sensors_py.Basic_Strip_Sensor),
    ]
    
    strip_width = 512
    frame_heigth = 512
    strip_height = 16
    strip_stride = 16
    strip_count = ((frame_heigth - strip_height) // strip_stride) + 1
    print("strip_count:", strip_count)
    
    # Alt. 8 8 64
    
    for video_debug_name, video_fname, reference_frame_fname, frame_range in videos:
        
        stream = Video_Stream(cv2.VideoCapture(video_fname))
        reference_frame = load_image(reference_frame_fname)
        '''
        print(stream.get_num_frames())
        plt.imshow(stream.get_frame(0))
        plt.show()
        '''
        
        if frame_range is None:
            frame_range = range(stream.get_num_frames())
        
        for sensor_name, sensor_class in sensors:
            sensor = sensor_class(reference_frame, (strip_height, strip_width))
            
            sensor_match_locs = []
            sensor_uncertainties = []
            sensor_extra_data = {}
            
            def export_data(sensor_match_locs, sensor_uncertainties, sensor_extra_data):
                # Export data
                sensor_match_locs = np.array(substitute_none_with_nan(sensor_match_locs))
                sensor_uncertainties = np.array(substitute_none_with_nan(sensor_uncertainties))
                
                print("Exporting data...")
                print(sensor_match_locs.shape)
                print(sensor_uncertainties.shape)
                np.save('{}_{}_locs.npy'.format(video_debug_name, sensor_name), sensor_match_locs)
                np.save('{}_{}_uncertainty.npy'.format(video_debug_name, sensor_name), sensor_uncertainties)
                
                for extra_data_name in sensor_extra_data:
                    sensor_extra = sensor_extra_data[extra_data_name]
                    sensor_extra = np.array(substitute_none_with_nan(sensor_extra))
                    np.save('{}_{}_{}.npy'.format(video_debug_name, sensor_name, extra_data_name), sensor_extra)
            
            for frame_idx in frame_range:
                frame = stream.get_frame(frame_idx)
                for strip_idx_in_frame in range(strip_count):
                    
                    strip_global_idx = frame_idx * strip_count + strip_idx_in_frame
                    
                    strip_data = frame[strip_idx_in_frame*strip_stride:strip_idx_in_frame*strip_stride+strip_height,:]
                    
                    result = sensor.sense_extra(strip_data)
                    if result is not None:
                        sensor_match_loc, sensor_uncertainty, sensor_extra = result
                    else:
                        sensor_match_loc, sensor_uncertainty, sensor_extra = (None, None, {})
                        
                    # account for position of strip in frame
                    sensor_match_loc -= np.array([strip_idx_in_frame*strip_stride, 0])
                    
                    for extra_data_name in sensor_extra:
                        extra_data = sensor_extra[extra_data_name]
                        if extra_data_name not in sensor_extra_data:
                            sensor_extra_data[extra_data_name] = []
                        sensor_extra_data[extra_data_name].append(extra_data)
                        
                        
                    sensor_match_locs.append(sensor_match_loc)
                    sensor_uncertainties.append(sensor_uncertainty)
                    
                    print(frame_idx, strip_global_idx, sensor_match_loc)
                    
                if frame_idx % 20 == 0:
                    # Save checkpoint
                    export_data(sensor_match_locs, sensor_uncertainties, sensor_extra_data)
                    
            export_data(sensor_match_locs, sensor_uncertainties, sensor_extra_data)
            

if __name__ == '__main__':
    main()
    
    
    
    
    
    
