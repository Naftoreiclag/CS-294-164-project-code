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
    plt.show()


def main():
    
    video_names = ['test1']
    sensor_name = 'Basic'
    standard_dev_threshold = 2
    max_distance_threshold = 100
    median_window_size = 5
    strips_per_frame = 32
    
    threshold = 0.0
    
    cleans = []
    cleans_min_bias = []
    
    results = {}
    for threshold in [0.1]:
        for video_debug_name in video_names:
            locs_fname = '{}_{}_locs.npy'.format(video_debug_name, sensor_name)
            uncertainty_fname = '{}_{}_uncertainty.npy'.format(video_debug_name, sensor_name)
            peaks_fname = '{}_{}_peak_size.npy'.format(video_debug_name, sensor_name)
            sensor_match_locs = np.load(locs_fname)
            sensor_uncertainties = np.load(uncertainty_fname)
            sensor_peaks = np.load(peaks_fname)
            
            cleaned = sensor_match_locs.copy()
            cleaned = cleaned.astype(np.float64)
            sel = sensor_peaks < threshold
            sel = np.hstack((sel, sel))
            cleaned[sel] = np.nan
            
            results[threshold] = cleaned
            
    dhs_fname = '{}_{}_locs_rbr.npy'.format(video_debug_name, "DHS")
    dhs_data = np.load(dhs_fname)
            
    show(results)
    

if __name__ == '__main__':
    main()
