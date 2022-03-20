from this import d
import numpy as np

class WeightDecayScheduler():
    def __init__(self):
        pass
    
    def __call__(epoch):
        if epoch < 8:
            decay = 1e-5
        elif epoch < 16:
            decay = 1e-4
        elif epoch < 20:
            decay = 5e-4
        elif epoch < 28:
            decay = 1e-3
        elif epoch < 36:
            decay = 5e-3
        elif epoch < 50:
            decay = 1e-2
        
        return decay


class LrDecayScheduler():
    def __init__(self, base_lr=0.0001, decay=0.9, step=1):
        self.step  = step
        self.decay = decay
        self.base_lr = base_lr

    def __call__(self, epoch):
        lr = self.base_lr * (self.decay**(epoch // self.step))
        return lr


class LrRampScheduler:
    def __init__(self,
                lr_start=0.0000005,
                lr_max=0.000005,
                lr_min=0.000001,
                lr_ramp_ep=5,
                lr_sus_ep=0,
                lr_decay=0.9):
        
        
        self.lr_start = lr_start
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.lr_ramp_ep = lr_ramp_ep
        self.lr_sus_ep = lr_sus_ep
        self.lr_decay = lr_decay
    
    def __call__(self, epoch):
        if epoch < self.lr_ramp_ep:
            lr = (self.lr_max - self.lr_start) / self.lr_ramp_ep * epoch + self.lr_start
            
        elif epoch < self.lr_ramp_ep + self.lr_sus_ep:
            lr = self.lr_max
            
        else:
            lr = (self.lr_max - self.lr_min) * self.lr_decay**(epoch - self.lr_ramp_ep - self.lr_sus_ep) + self.lr_min
            
        return lr

class ResolutionScheduler:
    def __init__(self, min_res=256, max_res=512, start_ramp=4, end_ramp=15):
        assert min_res <= max_res
        
        self.min_res = min_res
        self.max_res = max_res
        self.start_ramp = start_ramp
        self.end_ramp = end_ramp
    
    def __call__(self, epoch):
        if epoch < self.start_ramp:
            return self.min_res

        if epoch > self.end_ramp:
            return self.max_res
        
        # otherwise, linearly interpolate to the nearest multiple of 32
        interp = np.interp([epoch], [self.start_ramp, self.end_ramp], [self.min_res, self.max_res])
        res = int(np.round(interp[0] / 32)) * 32
        return res



if __name__ == '__main__':
    pass