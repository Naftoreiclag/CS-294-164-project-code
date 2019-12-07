#!/usr/bin/env python
import numpy as np
from filters import Kalman, Zero_Velocity, First_Order_Velocity_Approx

def apply_tracking_filter(filt, sensor_match_locs, sensor_uncertainties, latency):
    
    retval = []
    
    # Append enough to account for warm-up
    retval.extend([None] * latency)
    
    for step_idx in range(len(sensor_match_locs) - latency):
        sensor_reading = sensor_match_locs[step_idx]
        sensor_uncertainty = sensor_uncertainties[step_idx]
        
        filt.update(step_idx, sensor_reading, sensor_uncertainty)
        
        prediction = filt.predict(step_idx + latency)[0]
        retval.append(prediction)
        
    return np.array(substitute_none_with_nan(retval))

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
    
def main():
    videos = ['test1']
    sensors = ['DHS', 'Downsample', 'Basic']
    filters = [
        ('Kalman', Kalman),
        ('Zerovel', Zero_Velocity),
        ('FOVel', First_Order_Velocity_Approx)
    ]
    latencies = [0,1,2,3,4,5,6,7,8,9,10]
    
    particle_dyn = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, .9, 0],
        [0, 0, 0, .9],
    ], dtype=np.float64)
    particle_rand = np.array([
        [20, 0, 0, 0],
        [0, 20, 0, 0],
        [0, 0, 5, 0],
        [0, 0, 0, 5],
    ], dtype=np.float64)
    sensor_matr = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float64)
    init_particle_x = np.array([0, 0, 0, 0], dtype=np.float64)
    init_particle_uncertainty = np.eye(4, 4, dtype=np.float64)
    
    for video_debug_name in videos:
        for sensor_name in sensors:
            locs_fname = '{}_{}_locs_reg.npy'.format(video_debug_name, sensor_name)
            uncertainty_fname = '{}_{}_uncertainty.npy'.format(video_debug_name, sensor_name)
            
            sensor_match_locs = np.load(locs_fname)
            
            try:
                sensor_uncertainties = np.load(uncertainty_fname)
            except FileNotFoundError:
                print("Generating identity matrices. No uncertainties found at {}".format(uncertainty_fname))
                sensor_uncertainties = np.array([np.eye(sensor_match_locs[0].shape[0])] * len(sensor_match_locs))
            
            for filt_name, filter_class in filters:
                for latency in latencies:
                    filt = filter_class(init_particle_x, init_particle_uncertainty, particle_dyn, sensor_matr, particle_rand)
                    
                    prediction = apply_tracking_filter(filt, sensor_match_locs, sensor_uncertainties, latency)
                    
                    output_fname = '{}_{}_{}_{}_latency'.format(video_debug_name, sensor_name, filt_name, latency)
                    print(output_fname)
                    np.save(output_fname, prediction)
                    
    
if __name__ == '__main__':
    main()
    
