#!/usr/bin/env python
import numpy as np

class Random_Walk_Params:
    def __init__(self, particle_x, particle_dyn, particle_rand, num_steps, seed=None):
        
        if seed is None:
            seed = np.random.randint(1, 1e8)
        
        self.seed = seed
        self.particle_x = particle_x
        self.particle_dyn = particle_dyn
        self.particle_rand = particle_rand
        self.num_steps = num_steps
        
class Simulated_Sensor_Params:
    def __init__(self, sensor_matr, sensor_uncertainty, seed=None):
        
        if seed is None:
            seed = np.random.randint(1, 1e8)
            
        self.seed = seed
        self.sensor_matr = sensor_matr
        self.sensor_uncertainty = sensor_uncertainty

def generate_ground_truth_from_random_walk(random_walk_params):
    rwp = random_walk_params
    np.random.seed(rwp.seed)
    
    retval = []
    
    particle_x = np.copy(rwp.particle_x)
    
    for _ in range(rwp.num_steps):
        retval.append(particle_x)
        particle_x = rwp.particle_dyn @ particle_x
        particle_x += np.random.multivariate_normal(np.zeros((rwp.particle_rand.shape[0], )), rwp.particle_rand)
    
    return retval
    
def generate_sensor_data_from_ground_truth(simulated_sensor_params, ground_truth):
    ssp = simulated_sensor_params
    np.random.seed(ssp.seed)
    
    retval = []
    
    for particle_x in ground_truth:
        if particle_x is None:
            sensor_reading = None
        else:
            sensor_reading = ssp.sensor_matr @ particle_x
            sensor_reading += np.random.multivariate_normal(np.zeros((ssp.sensor_uncertainty.shape[0], )), ssp.sensor_uncertainty)
            
        sensor_uncertainty = np.copy(ssp.sensor_uncertainty)
        
        
        retval.append((sensor_reading, sensor_uncertainty))
        
    return retval
