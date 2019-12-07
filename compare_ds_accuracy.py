#!/usr/bin/env python
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.stats import iqr
         
def remove_nan_rows(arr):
    return arr[~np.isnan(arr)]
    
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
    
def get_standard_dev(a, b):
    
    total_len = min(a.shape[0], b.shape[0])
    a = a[:total_len]
    b = b[:total_len]
    
    err = a-b
    err = remove_nan_rows(err)
    
    return np.std(err, axis=0)
    
def show_results(time_series):
    
    arbitrary = None
    for key in time_series:
        arbitrary = time_series[key]
    
    num_steps = len(arbitrary)
    
    fig, axs = plt.subplots()
    xs = list(range(num_steps))
    
    for test_name  in time_series:
        data = time_series[test_name]
        data = substitute_none_with_nan(data)
        data = np.array(data)[:num_steps]
        
        #print(test_name, get_standard_dev(ground_truth, data))
        
        axs.plot(xs, data, '.-', label=test_name)
    
    axs.legend()
    plt.show()
    


def main():
    videos = ['test1']
    downsample = 'Downsample'
    compare_against = [
        ('DHS', 'DHS'), 
        ('Basic', 'WSN'), 
    ]
    
    def iqr_err(error):
        metric = iqr(error, axis=0, nan_policy='omit')
        return metric
    def std_err(error):
        metric = np.nanstd(error, axis=0)
        return metric
    
    for video_debug_name in videos:
        downsample_fname = '{}_{}_locs_reg.npy'.format(video_debug_name, downsample)
        downsample_data = np.load(downsample_fname)
        
        for sensor_name, _ in compare_against:
            other_fname = '{}_{}_locs_reg.npy'.format(video_debug_name, sensor_name)
            other_data = np.load(other_fname)
            
            diff = downsample_data - other_data
            diff = np.linalg.norm(diff, axis=1)
            
            
            
            print("{}:\t{}\t{}\t{}".format(sensor_name, iqr_err(diff), std_err(diff), np.nanmax(diff)))
            

if __name__ == '__main__':
    main()
