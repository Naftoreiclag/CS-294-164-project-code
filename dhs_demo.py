#!/usr/bin/env python
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

def color_ramp(x, ramp='rainbow'):
    scalar_map = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=1), cmap=plt.get_cmap(ramp))
    y = scalar_map.to_rgba(x)
    return np.array(y[:3])

def main():
    # List of videos and the corresponding reference, number of frames max to use
    videos = [
        ('test1', 'test_data/20160R_V006.avi', 'test_data/20160R_V006_Reference.png', range(105,300))
    ]
    
    strip_width = 512
    strip_height = 16
    strip_stride = 16
    strip_count = 32
    
    for video_debug_name, video_fname, reference_frame_fname, frame_range in videos:
        
        stream = Video_Stream(cv2.VideoCapture(video_fname))
        reference_frame = load_image(reference_frame_fname)
        
        if frame_range is None:
            frame_range = range(stream.get_num_frames())
    
        num_partitions = 2
        sensor = sensors_py.Double_Half_Strip(reference_frame, (strip_height, strip_width), num_partitions)
        
        for frame_idx in frame_range:
            frame = stream.get_frame(frame_idx)
            for strip_idx_in_frame in range(strip_count):
                
                strip_global_idx = frame_idx * strip_count + strip_idx_in_frame
                
                strip_data = frame[strip_idx_in_frame*strip_stride:strip_idx_in_frame*strip_stride+strip_height,:]
                
                peak_mean, peak_covar, peak_raw, peak_sizes = sensor.sense_with_metadata(strip_data)
                
                std_dev = np.sqrt(np.sum(peak_covar))
                print('frame {}, strip {}'.format(frame_idx, strip_idx_in_frame), std_dev)
                if True:
                    print('peaks: ', peak_raw)
                    print('peak_size: ', peak_sizes)
                    fig, axs = plt.subplots(1,2,figsize=(15,6))
                    leftax = axs[0]
                    rightax = axs[1]
                    leftax.imshow(frame, cmap='gray')
                    rightax.imshow(reference_frame, cmap='gray')
                    for i in range(num_partitions):
                        rect = patches.Rectangle(np.flip(peak_raw[i]), strip_width, strip_height, linewidth=1, edgecolor=color_ramp(i/num_partitions), facecolor='none')
                        rightax.add_patch(rect)
                    rect = patches.Rectangle((0,strip_idx_in_frame*strip_stride), strip_width, strip_height, linewidth=1, edgecolor=color_ramp(1), facecolor='none')
                    leftax.add_patch(rect)
                    plt.show()
                

if __name__ == '__main__':
    main()
    
    
    
    
    
    
