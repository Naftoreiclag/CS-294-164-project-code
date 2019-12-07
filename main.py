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
        
        axs.plot(xs, data, '.-', label=test_name)
    
    axs.legend()
    plt.show()
    


def main():
    videos = ['test1']
    sensors = [
        ('DHS', 'DHS'), 
        ('Basic', 'WSN'), 
        ('Downsample', 'DS'),
    ]
    filters = [
        ('Kalman', 'KAL'), 
        ('Zerovel', 'ZV'), 
        ('FOVel', 'FV'),
    ]
    latencies = [0,1,2,3,4,5,6,7,8,9,10]
    
    def exclude_kalman_for_basic_and_downsample(key):
        video_debug_name, sensor_name, filt_name, latency = key
        
        return filt_name == 'Kalman' and sensor_name in ['Downsample', 'Basic']
    
    export_blacklist_fns = [
        exclude_kalman_for_basic_and_downsample
    ]
    
    for video_debug_name in videos:
        all_results = {}
        all_errors = {}
        for sensor_name, _ in sensors:
            truth_fname = '{}_{}_locs_reg.npy'.format(video_debug_name, sensor_name)
            ground_truth = np.load(truth_fname)
            for filt_name, _ in filters:
                for latency in latencies:
                    predictions_fname = '{}_{}_{}_{}_latency.npy'.format(video_debug_name, sensor_name, filt_name, latency)
                    predictions = np.load(predictions_fname)
                    
                    error = ground_truth - predictions[:,:2]
                    error = np.linalg.norm(error, axis=1)
                    
                    key = (video_debug_name, sensor_name, filt_name, latency)
                    
                    all_results[key] = predictions
                    all_errors[key] = error
        
        output_data_table = []
        
        def iqr_err(error):
            metric = iqr(error, axis=0, nan_policy='omit')
            return metric
        def std_err(error):
            metric = np.nanstd(error, axis=0)
            return metric
        
        wrote_header = False
        header_data = []
        pretty_header_data = []
        header_data.append('latency')
        pretty_header_data.append('Latency')
        
        for latency in latencies:
            
            row_data = []
            row_data.append('{}'.format(latency))
            
            for metric_fn, metric_fn_name in [(iqr_err, 'IQR'), (std_err, 'STD')]:
                for sensor_name, senor_name_abbr in sensors:
                    for filt_name, filt_name_abbr in filters:
                        key = (video_debug_name, sensor_name, filt_name, latency)
                        
                        if any(fn(key) for fn in export_blacklist_fns):
                            continue
                        
                        error = all_errors[key]
                        
                        metric = metric_fn(error)
                        metric = '{:.2f}'.format(metric)
                        
                        row_data.append(metric)
                        if not wrote_header:
                            header_data.append('{}_{}_{}'.format(sensor_name, filt_name, metric_fn_name))
                            pretty_header_data.append('{} +{}'.format(senor_name_abbr, filt_name_abbr))
                      
            if not wrote_header:
                output_data_table.append(header_data)
                output_data_table.append(pretty_header_data)
                wrote_header = True
            output_data_table.append(row_data)
        
        data_table_output_fname = '{}_accuracy_results'.format(video_debug_name)
        with open(data_table_output_fname + '.csv', 'w', newline='') as ofile:
            writer = csv.writer(ofile)
            for row in output_data_table:
                writer.writerow(row)
                
        with open(data_table_output_fname + '.tex', 'w') as ofile:
            for row in output_data_table:
                ofile.write('&'.join(row) + '\\\\\n')
                    
        
        print('\n'.join(str(x) for x in output_data_table))    
        print(data_table_output_fname)
        

if __name__ == '__main__':
    main()
