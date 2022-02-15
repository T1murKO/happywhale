
def ramp_scheduler(num_epoch, batch_size, start_epoch=0):
    lr_start   = 0.000001
    lr_max     = 0.000005 * batch_size
    lr_min     = 0.000001
    lr_ramp_ep = 4
    lr_sus_ep  = 0
    lr_decay   = 0.92
    
    def get_lr(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
            
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
            
        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
            
        return lr
    
    
    lr_schedule = [get_lr(epoch_n)for epoch_n in range(start_epoch, num_epoch, 1)]
    
    return lr_schedule