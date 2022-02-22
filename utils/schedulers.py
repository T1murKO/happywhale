
def ramp_scheduler(batch_size,
                   lr_start=0.0000005,
                   lr_max=0.000005,
                   lr_min=0.000001,
                   lr_ramp_ep=5,
                   lr_sus_ep=0,
                   lr_decay=0.9):

    lr_start   = lr_start
    lr_max     = lr_max * batch_size
    lr_min     = lr_min
    lr_ramp_ep = lr_ramp_ep
    lr_sus_ep  = lr_sus_ep
    lr_decay   = lr_decay
    
    def get_lr(epoch):
        
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
            
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
            
        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
            
        return lr
    
    return get_lr


if __name__ == '__main__':
    scheduler = ramp_scheduler(10, 32)
    
    for i in range(20):
        print(scheduler(i))