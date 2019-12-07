#!/usr/bin/env python
import numpy as np

class Filter:
    def predict(self, timestep):
        raise NotImplementedError()
        
    def update(self, timestep, zvec, rmatr):
        raise NotImplementedError()
    
class Zero_Velocity(Filter):
    '''
    "Bad" prediction technique: just assume no eye motion.
    '''
    
    def __init__(self, init_xhat, init_pcov, fmatr, hmatr, qmatr):
        self.init_xhat = init_xhat.astype(np.float64)
        self.current_x_pred = np.copy(self.init_xhat)
        
    def predict(self, timestep):
        timestep = int(timestep)
        
        assert(self.current_timestep <= timestep)
        
        return np.copy(self.current_x_pred), np.eye(self.current_x_pred.shape[0])
        
    def update(self, timestep, zvec, rmatr):
        # Exclude any nan's
        if np.any(np.isnan(zvec)) or np.any(np.isnan(rmatr)):
            return
        
        timestep = int(timestep)
        self.current_timestep = timestep
        self.current_x_pred = np.copy(zvec)
    
class First_Order_Velocity_Approx(Filter):
    '''
    Use the previous data point to approximate velocity
    '''
    
    def __init__(self, init_xhat, init_pcov, fmatr, hmatr, qmatr):
        self.init_xhat = np.copy(init_xhat).astype(np.float64)
        
        self.prev_x = hmatr @ self.init_xhat
        self.prev_x_time = -2
        self.current_x = np.copy(self.prev_x)
        self.current_x_time = -1
        
        self.current_timestep = -1
        
    def predict(self, timestep):
        timestep = int(timestep)
        
        assert(self.current_timestep <= timestep)
        
        # Get the slope
        delta = self.current_x - self.prev_x
        delta /= self.current_x_time - self.prev_x_time
        
        # Perform prediction
        prediction = self.current_x + (delta * (timestep - self.current_x_time))
        
        return prediction, np.eye(prediction.shape[0])
        
    def update(self, timestep, zvec, rmatr):
        # Exclude any nan's
        if np.any(np.isnan(zvec)) or np.any(np.isnan(rmatr)):
            return
        
        assert(self.current_timestep < timestep)
        
        timestep = int(timestep)
        
        self.prev_x = self.current_x
        self.prev_x_time = self.current_x_time
        
        self.current_x = np.copy(zvec).astype(np.float64)
        self.current_x_time = timestep
        
        self.current_timestep = timestep

class Kalman(Filter):
    '''
    Kalman filter implementation.
    There is no control term.

    Assumes that the prediction matrix (fmatr) is constant for all timesteps
    Assumes that the sensor matrix (hmatr) is constant for all timesteps
    Assumes that the noise matrix (qmatr) is constant for all timesteps
    '''
    
    def __init__(self, init_xhat, init_pcov, fmatr, hmatr, qmatr):
        
        init_xhat = init_xhat.astype(np.float64)
        init_pcov = init_pcov.astype(np.float64)
        fmatr = fmatr.astype(np.float64)
        hmatr = hmatr.astype(np.float64)
        qmatr = qmatr.astype(np.float64)
        
        # Timestep currently held in xhat and pcov
        self.current_timestep = 0
        self.xhat = np.copy(init_xhat)
        self.pcov = np.copy(init_pcov)
        
        # The prediction, sensor, noise matrices
        self.fmatr = np.copy(fmatr)
        self.hmatr = np.copy(hmatr)
        self.qmatr = np.copy(qmatr)
        
    def predict(self, timestep):
        '''
        Returns the predicted mean and covariance of the state at time timestep
        '''
        timestep = int(timestep)
        
        assert(self.current_timestep <= timestep)
        
        ret_xhat = np.copy(self.xhat)
        ret_pcov = np.copy(self.pcov)
        
        for _ in range(timestep - self.current_timestep):
            ret_xhat = self.fmatr @ ret_xhat
            ret_pcov = self.fmatr @ ret_pcov @ self.fmatr.T + self.qmatr
            
        return ret_xhat, ret_pcov
        
    
    def update(self, timestep, zvec, rmatr):
        # Exclude any nan's
        if np.any(np.isnan(zvec)) or np.any(np.isnan(rmatr)):
            return
        
        zvec = zvec.astype(np.float64)
        rmatr = rmatr.astype(np.float64)
        
        '''
        Updates our estimate given the sensor reading.
        Requires that timestep increases each time this is called.
        '''
        timestep = int(timestep)
        
        # First, we predict the given timestep
        
        self.xhat, self.pcov = self.predict(timestep)
        self.current_timestep = timestep
        
        # Then, we apply the update onto the current estimate.
        
        pkhkt = self.pcov @ self.hmatr.T
        
        kalman_gain = pkhkt @ np.linalg.inv((self.hmatr @ self.pcov @ self.hmatr.T) + rmatr) # 19
        
        self.xhat += kalman_gain @ (zvec - (self.hmatr @ self.xhat)) # 18
        self.pcov -= kalman_gain @ (self.hmatr @ self.pcov) # 18
        
