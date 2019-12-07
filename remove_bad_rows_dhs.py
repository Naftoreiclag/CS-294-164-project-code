#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math

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

def show(trace):
    fig, axs = plt.subplots()
    xs = list(range(len(trace)))
    axs.plot(xs, trace, '.-', label="trace")
    axs.legend()
    plt.show()

def show_compare(trace1,trace2):
    fig, axs = plt.subplots()
    xs = list(range(len(trace1)))
    axs.plot(xs, trace1, '.-', label="trace1")
    axs.plot(xs, trace2, '.-', label="trace2")
    axs.legend()
    plt.show()
    
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
        else:
            # Check if we are within the median
            median = median_with_nan(retval[-median_window_size:])
            if median is not None and np.linalg.norm(median - sensor_match) > max_distance_threshold:
                fail_dist += 1
                all_tests_passed = False
        
        if all_tests_passed:
            retval.append(sensor_match)
            
            #if sensor_match[0] > 200:
            #    print(global_idx % 32, global_idx // 32 )
            #    print(sensor_match)
                
        else:
            retval.append(None)
        
        #if global_idx % 32 == 31:
        #    retval.append(None)
        
    retval = substitute_none_with_nan(retval)
    
    return np.array(retval)

def bias_analyizer(strip_locations, strips_per_frame, show_figs=False):
    
    total_num_matches = strip_locations.shape[0]
    dimensions = strip_locations.shape[1]
    assert(total_num_matches % strips_per_frame == 0)
    
    per_frame_data = np.reshape(strip_locations, (total_num_matches // strips_per_frame, strips_per_frame, dimensions))
    
    per_frame_deltas = []
    
    for i in range(len(per_frame_data)):
        frame = per_frame_data[i]
        
        deltas = []
        for j in range(len(frame)-1):
            deltas.append(frame[j+1] - frame[j])
        
        per_frame_deltas.append(deltas)
        
    per_frame_deltas = np.array(per_frame_deltas)
    print(per_frame_deltas.shape)
    
    raw_medians = np.nanmedian(per_frame_deltas, axis=0)
    raw_averages = np.nanmean(per_frame_deltas, axis=0)
    q1_percentile = np.nanpercentile(per_frame_deltas, q=25, axis=0)
    q3_percentile = np.nanpercentile(per_frame_deltas, q=75, axis=0)
    iqrs = (q3_percentile - q1_percentile) + 0.5
    tukey_lower = q1_percentile - 1.5 * iqrs
    tukey_upper = q3_percentile + 1.5 * iqrs
    per_frame_deltas_filtered = np.copy(per_frame_deltas)
    per_frame_deltas_filtered[per_frame_deltas < tukey_lower] = np.nan
    per_frame_deltas_filtered[per_frame_deltas > tukey_upper] = np.nan
    tukey_fenced_averages = np.nanmean(per_frame_deltas_filtered, axis=0)
        
    
    # dim 0 is the frame idx
    # dim 1 is the strip idx in the frame
    # dim 2 is x/y
    
    # Reconstruct the reference bias
    reference_bias = []
    reference_bias.append(np.zeros(tukey_fenced_averages[0].shape))
    for i in range(tukey_fenced_averages.shape[0]):
        reference_bias.append(reference_bias[i] + tukey_fenced_averages[i])
    reference_bias = np.array(reference_bias)
    
    def tukey_averge(arr):
        q1_percentile = np.nanpercentile(arr, q=25, axis=0)
        q3_percentile = np.nanpercentile(arr, q=75, axis=0)
        iqrs = q3_percentile - q1_percentile
        tukey_lower = q1_percentile - 1.5 * iqrs
        tukey_upper = q3_percentile + 1.5 * iqrs
        arr_filtered = np.copy(arr)
        arr_filtered[arr < tukey_lower] = np.nan
        arr_filtered[arr > tukey_upper] = np.nan
        tukey_fenced_averages = np.nanmean(arr_filtered, axis=0)
        return tukey_fenced_averages
        
    
    def show_bias():
        fig, axs = plt.subplots(1, 2)
        xs = list(range(per_frame_data.shape[1]))
        
        asdf = []
        for i in range(0, per_frame_data.shape[0]):
            asdf.append(per_frame_data[i][:] - per_frame_data[i][0])
        asdf = np.array(asdf)
        
        asdf_avg = np.nanmedian(asdf, axis=0)
        asdf_tukey = tukey_averge(asdf)
        
        for i in range(1, per_frame_data.shape[0]):
            axs[0].plot(xs, asdf[i][:,0] - reference_bias[:,0], '-', color=(0, 0, 0, 0.5))
            axs[1].plot(xs, asdf[i][:,0], '-', color=(0, 0, 0, 0.5))
        axs[1].plot(xs, reference_bias[:,0], 'o-')
        axs[1].plot(xs, asdf_avg[:,0], 'x-')
        axs[1].plot(xs, asdf_tukey[:,0], '.-')
            
        plt.show()
    
    def show_deltas():
        fig, axs = plt.subplots()
        xs = list(range(per_frame_deltas.shape[1]))
        
        for i in range(1, per_frame_deltas.shape[0]):
            axs.plot(xs, per_frame_deltas[i][:,0], '.')
        axs.plot(xs, tukey_fenced_averages[:,0], 'o-')
        axs.plot(xs, raw_averages[:,0], 'o-')
        axs.plot(xs, tukey_lower[:,0])
        axs.plot(xs, tukey_upper[:,0])
        #axs.legend()
            
        plt.show()
    
    def show_hist():
        total = per_frame_deltas.shape[1]
        width = math.ceil(total ** 0.5)
        
        fig, axs = plt.subplots(width, width, tight_layout=True, sharey=True)
        for i in range(per_frame_deltas.shape[1]):
            axs[i//width,i%width].hist(per_frame_deltas[:,i,0])
        plt.show()

    if show_figs:
        #show_hist()
        show_deltas()
        show_bias()
    
    return reference_bias
    

def load_clean_data(video_debug_name, sensor_name, standard_dev_threshold, max_distance_threshold, median_window_size):
    locs_fname = '{}_{}_locs.npy'.format(video_debug_name, sensor_name)
    uncertainty_fname = '{}_{}_uncertainty.npy'.format(video_debug_name, sensor_name)
    
    sensor_match_locs = np.load(locs_fname)
    sensor_uncertainties = np.load(uncertainty_fname)
    
    cleaned = clean_up_data(sensor_match_locs, sensor_uncertainties, standard_dev_threshold, max_distance_threshold, median_window_size)
    
    save_data = False
    if save_data:
        output_fname = '{}_{}_locs_bad_rem.npy'.format(video_debug_name, sensor_name)
        np.save(output_fname, cleaned)
        
    return cleaned
    
    
def subtract_bias(strip_locations, reference_bias):
    
    strips_per_frame = reference_bias.shape[0]
    total_num_matches = strip_locations.shape[0]
    assert(total_num_matches % strips_per_frame == 0)
    
    rep_times = total_num_matches // strips_per_frame
    
    
    reference_copied = np.tile(reference_bias, (rep_times, 1))
    
    return strip_locations - reference_copied

def main():
    
    video_names = ['test1']
    sensor_name = 'DHS'
    standard_dev_threshold = 2
    max_distance_threshold = 100
    median_window_size = 5
    strips_per_frame = 32
    
    cleans = []
    cleans_min_bias = []
    
    
    for video_debug_name in video_names:
        locs_fname = '{}_{}_locs.npy'.format(video_debug_name, sensor_name)
        uncertainty_fname = '{}_{}_uncertainty.npy'.format(video_debug_name, sensor_name)
        peaks_fname = '{}_{}_uncertainty.npy'.format(video_debug_name, sensor_name)
    
        cleaned = load_clean_data(video_debug_name, sensor_name, standard_dev_threshold, max_distance_threshold, median_window_size)
        
        
        bias = bias_analyizer(cleaned, strips_per_frame)
        
        cleaned_no_bias = subtract_bias(cleaned, bias)
        
        cleans.append(cleaned)
        cleans_min_bias.append(cleaned_no_bias)
        
        
        np.save('{}_{}_locs_rbr.npy'.format(video_debug_name, sensor_name), cleaned)
        np.save('{}_{}_locs_rbrnb.npy'.format(video_debug_name, sensor_name), cleaned_no_bias)
        
    #test = cleans[1][:strips_per_frame]
    show(cleans_min_bias[0])
    
    #show(subtract_bias(cleans[0], -test) -cleans[1])
    #show((cleans_min_bias[0] - np.nanmean(cleans_min_bias[0], axis=0)) - (cleans_min_bias[1] - np.nanmean(cleans_min_bias[1], axis=0)))
    

if __name__ == '__main__':
    main()
