#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math

def show(traces):
    fig, axs = plt.subplots()
    
    for trace_name in traces:
        trace = traces[trace_name]
        xs = list(range(len(trace)))
        axs.plot(xs, trace[:,1], '.-', label=str(trace_name))
    axs.set_ylim(-100, 200)
    axs.set_xlim(5500, 5600)
    #axs.legend()
    plt.show()

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

def clean_up_data_using_threshold(sensor_match_locs, sensor_peaks, threshold):
    cleaned = sensor_match_locs.copy()
    cleaned = cleaned.astype(np.float64)
    sel = sensor_peaks < threshold
    sel = np.hstack((sel, sel))
    cleaned[sel] = np.nan
    
    return cleaned
    
def median_with_nan(lst):
    x = []
    for item in lst:
        if item is not None:
            x.append(item)
    if len(x) == 0:
        return None
    else:
        return np.median(np.array(x), axis=0)

def clean_up_data(sensor_match_locs, sensor_uncertainties, std_dev_thresh, max_dist_thresh, max_dist_window_size):
    retval = []
    fail_dist = 0
    fail_std = 0
    standard_dev_threshold = std_dev_thresh
    max_distance_threshold = max_dist_thresh
    median_window_size = max_dist_window_size
    for global_idx, (sensor_match, sensor_uncertainty) in enumerate(zip(sensor_match_locs, sensor_uncertainties)):
        total_std = np.sqrt(np.sum(sensor_uncertainty))
        
        all_tests_passed = True
        
        # Check if we exceed the threshold
        if total_std > standard_dev_threshold:
            fail_std += 1
            all_tests_passed = False
        # Check if we are within the median
        median = median_with_nan(retval[-median_window_size:])
        if median is not None and np.linalg.norm(median - sensor_match) > max_distance_threshold:
            fail_dist += 1
            all_tests_passed = False
        
        if all_tests_passed:
            retval.append(sensor_match)
        else:
            retval.append(None)
        
    retval = substitute_none_with_nan(retval)
    
    return np.array(retval)

def main():
    
    video_names = ['test1']
    sensor_name = 'Downsample'
    standard_dev_threshold = 2
    max_distance_threshold = 100
    median_window_size = 5
    strips_per_frame = 32
    
    std_dev_thresh = 9999
    max_dist_thresh = 100
    max_dist_window_size = 5
    peak_threshold = 0.0
    
    cleans = []
    cleans_min_bias = []
    
    results = {}
    for video_debug_name in video_names:
        locs_fname = '{}_{}_locs.npy'.format(video_debug_name, sensor_name)
        uncertainty_fname = '{}_{}_uncertainty.npy'.format(video_debug_name, sensor_name)
        sensor_match_locs = np.load(locs_fname)
        sensor_uncertainties = np.load(uncertainty_fname)
        
        try:
            peaks_fname = '{}_{}_peak_size.npy'.format(video_debug_name, sensor_name)
            sensor_peaks = np.load(peaks_fname)
            sensor_match_locs = clean_up_data_using_threshold(sensor_match_locs, sensor_peaks, peak_threshold)
        except FileNotFoundError:
            print("No peak thresholding. No peaks found at {}".format(peaks_fname))
        
        sensor_match_locs = clean_up_data(sensor_match_locs, sensor_uncertainties, std_dev_thresh, max_dist_thresh, max_dist_window_size)
        
        results[peak_threshold] = sensor_match_locs
            
    #dhs_fname = '{}_{}_locs_rbr.npy'.format(video_debug_name, "DHS")
    #dhs_data = np.load(dhs_fname)
    #results = {}
    #results["DHS"] = dhs_data
    
    
    #np.save('{}_{}_locs_rbr.npy'.format(video_debug_name, sensor_name), results[0.2])
            
    show(results)
    

if __name__ == '__main__':
    main()
