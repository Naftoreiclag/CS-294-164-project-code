#!/usr/bin/env python
'''
Kalman filter demo.

As far as I can tell, there is no agreed-upon standard for the different
variable names, so I just went with the names given here:
https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/
Which is a top Google search result.

xhat : estimation of the true particle state
pcov : particle state estimation uncertainty as a covariance matrix
fmatr : particle dynamics stored as a matrix
qmatr : noise added to the particle state as a multivariate gaussian with given covariance matrix
zvec : raw sensor reading
rmatr : sensor uncertainty as a covariance matrix
'''

import numpy as np
import matplotlib.pyplot as plt
import filters
import random_walk

def apply_tracking_filter(filt, sensor_data, latency):
    
    retval = []
    
    # Append enough to account for warm-up
    retval.extend([None] * latency)
    
    for step_idx, reading in enumerate(sensor_data):
        
        if reading is not None:
            sensor_reading, sensor_uncertainty = reading
            filt.update(step_idx, sensor_reading, sensor_uncertainty)
        
        prediction = filt.predict(step_idx + latency)[0]
        retval.append(prediction)
        
    return retval
    
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
    
def remove_nan_rows(arr):
    return arr[~np.isnan(arr).any(axis=1)]
    
def get_standard_dev(a, b):
    
    total_len = min(a.shape[0], b.shape[0])
    a = a[:total_len]
    b = b[:total_len]
    
    err = a-b
    err = remove_nan_rows(err)
    
    return np.std(err, axis=0)
    
    
def show_results(ground_truth, predictions):
    ground_truth = substitute_none_with_nan(ground_truth)
    num_steps = len(ground_truth)
    ground_truth = np.array(ground_truth)[:num_steps, :1]
    
    fig, axs = plt.subplots()
    xs = list(range(num_steps))
    axs.plot(xs, ground_truth, label="truth")
    
    for test_name  in predictions:
        prediction = predictions[test_name]
        prediction = substitute_none_with_nan(prediction)
        prediction = np.array(prediction)[:num_steps, :1]
        
        print(test_name, get_standard_dev(ground_truth, prediction))
        
        axs.plot(xs, prediction, label=test_name)
    
    axs.legend()
    plt.show()
    

def main():
    seed = np.random.randint(0, 1e6)
    print("seed", seed)
    np.random.seed(seed)
    num_steps = 100
    
    # Make random walk
    particle_x = np.array([0, 0, 0, 0], dtype=np.float64)
    particle_dyn = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, .9, 0],
        [0, 0, 0, .9],
    ], dtype=np.float64)
    particle_rand = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0.01, 0],
        [0, 0, 0, 0.01],
    ], dtype=np.float64)
    random_walk_params = random_walk.Random_Walk_Params(particle_x, particle_dyn, particle_rand, num_steps)
    
    # Make sensor
    sensor_matr = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float64)
    sensor_uncertainty = np.array([
        [0.0002, 0],
        [0, 0.0002],
    ], dtype=np.float64)
    simulated_sensor_params = random_walk.Simulated_Sensor_Params(sensor_matr, sensor_uncertainty)
    
    # Make filters
    init_particle_x = np.copy(particle_x)
    init_particle_uncertainty = np.eye(4, 4, dtype=np.float64)
    
    test_filters = {
        "kalman" : filters.Kalman(init_particle_x, init_particle_uncertainty, particle_dyn, sensor_matr, particle_rand),
        "zero_vel" : filters.Zero_Velocity(init_particle_x, init_particle_uncertainty, particle_dyn, sensor_matr, particle_rand)
    }
    latency = 3
    
    # Compute the "ground truth" sequence
    ground_truth = random_walk.generate_ground_truth_from_random_walk(random_walk_params)
    
    # Compute the sensor data from
    sensor_data = random_walk.generate_sensor_data_from_ground_truth(simulated_sensor_params, ground_truth)
    
    # Predict eye position
    
    predictions = {}
    for filt_name  in test_filters:
        filt = test_filters[filt_name]
        predicted = apply_tracking_filter(filt, sensor_data, latency)
        predictions[filt_name] = predicted
    
    # Display
    show_results(ground_truth, predictions)

    
if __name__ == '__main__':
    main()
        
        
        
        
        
