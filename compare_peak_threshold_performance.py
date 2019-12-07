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
    axs.legend()
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
        
def clean_up_data_using_std_dev_threshold(sensor_match_locs, sensor_uncertainties, std_dev_thresh):
    retval = []
    fail_std = 0
    standard_dev_threshold = std_dev_thresh
    for global_idx, (sensor_match, sensor_uncertainty) in enumerate(zip(sensor_match_locs, sensor_uncertainties)):
        total_std = np.sqrt(np.sum(sensor_uncertainty))
        
        all_tests_passed = True
        
        # Check if we exceed the threshold
        if total_std >= standard_dev_threshold:
            fail_std += 1
            all_tests_passed = False
        
        if all_tests_passed:
            retval.append(sensor_match)
        else:
            nothings = np.empty(sensor_match.shape)
            nothings[:] = np.nan
            retval.append(nothings)
    
    return np.array(retval)

def clean_up_data_using_infeasible_velocity(sensor_match_locs, max_dist_thresh, max_dist_window_size):
    retval = []
    fail_dist = 0
    max_distance_threshold = max_dist_thresh
    median_window_size = max_dist_window_size
    for global_idx, sensor_match in enumerate(sensor_match_locs):
        
        all_tests_passed = True
        
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

def remove_isolated_points(data):
    retval = data.copy()
    for item_idx in range(len(data)):
        # Check if left neighbor
        left_neighbor = item_idx == 0 or not np.any(np.isnan(data[item_idx - 1]))
        right_neighbor = item_idx == len(data) - 1 or not np.any(np.isnan(data[item_idx + 1]))
        
        if not (left_neighbor or right_neighbor):
            retval[item_idx] = np.nan
    return retval

def main():
    
    video_names = ['test1']
    sensor_names = ['Downsample', 'Basic', 'DHS']
    strips_per_frame = 32
    
    max_dist_thresh = 20
    max_dist_window_size = 5
    
    def count_rows(data):
        return np.sum(np.all(data, axis=1))
    
    for video_debug_name in video_names:
        for sensor_name in sensor_names:
            locs_fname = '{}_{}_locs.npy'.format(video_debug_name, sensor_name)
            sensor_match_locs = np.load(locs_fname)
            
            print('----- {}'.format(locs_fname))
            
            try:
                peaks_fname = '{}_{}_peak_size.npy'.format(video_debug_name, sensor_name)
                sensor_peaks = np.load(peaks_fname)
            except FileNotFoundError:
                print("No peaks found at {}".format(peaks_fname))
                sensor_peaks = None
            
            try:
                uncertainty_fname = '{}_{}_uncertainty.npy'.format(video_debug_name, sensor_name)
                sensor_uncertainties = np.load(uncertainty_fname)
            except FileNotFoundError:
                print("No uncertainty found at {}".format(peaks_fname))
                sensor_uncertainties = None
            
            if sensor_peaks is not None:
                param_search = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
                mode = 'peak'
            elif sensor_uncertainties is not None:
                param_search = [0, 1, 2, 3, 4, 5, 25, 50, 100, 200, 300, 350, 400, 450, 500][::-1]
                mode = 'stddev'
            else:
                param_search = None
                mode = None
                
            if not mode:
                print('No data for extra detection methods.')
                continue
            
            outliers_removed = clean_up_data_using_infeasible_velocity(sensor_match_locs, max_dist_thresh, max_dist_window_size)
            outliers_removed = remove_isolated_points(outliers_removed)
            total_data = len(sensor_match_locs)
            print('Total: {}'.format(total_data))
            is_lie = np.isnan(outliers_removed)
            is_truth = np.logical_not(is_lie)
            
            #show({'outs': outliers_removed})
            
            print('Lies: {}'.format(count_rows(is_lie)))
            print('Truth: {}'.format(count_rows(is_truth)))
            
            for parameter_val in param_search:
                
                if mode == 'peak':
                    after_thresholding = clean_up_data_using_threshold(sensor_match_locs, sensor_peaks, parameter_val)
                elif mode == 'stddev':
                    after_thresholding = clean_up_data_using_std_dev_threshold(sensor_match_locs, sensor_uncertainties, parameter_val)
                
                #if parameter_val == 1:
                #    show({'a': after_thresholding, 'b':outliers_removed})
                
                # Positives:
                # All rows where after_thresholding=None when sensor_match_locs!=None
                # Negatives:
                # All rows where after_thresholding!=None when sensor_match_locs!=None
            
                is_rejected = np.isnan(after_thresholding)
                is_kept = np.logical_not(is_rejected)
                
                # Outliers:
                # All rows where outliers_removed=None and sensor_match_locs!=None
                
            
                # True positives:
                # A non-outlier that is positive
                # False negatives:
                # An outlier that is negative
                
                true_positive = np.logical_and(is_rejected, is_lie) # Good: We rejected when we were supposed to
                true_negative = np.logical_and(is_kept, is_truth) # Good: We did not reject when we weren't
                false_positive = np.logical_and(is_rejected, is_truth) # Bad: We rejected when we weren't supposed to
                false_negative = np.logical_and(is_kept, is_lie) # Bad: We did not reject when we weren't supposed to
                
                confusion_matrix = np.array([
                    [count_rows(true_positive), count_rows(false_positive)],
                    [count_rows(false_negative), count_rows(true_negative)]
                ])
                
                
                print('====== threshold: {}'.format(parameter_val))
                
                print(confusion_matrix)
                #print('true_positive:\t{}'.format(count_rows(true_positive)))
                #print('true_negative:\t{}'.format(count_rows(true_negative)))
                #print('false_positive:\t{}'.format(count_rows(false_positive)))
                #print('false_negative:\t{}'.format(count_rows(false_negative)))

if __name__ == '__main__':
    main()
