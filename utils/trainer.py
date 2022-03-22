from sklearn.metrics import accuracy_score
import torch
import os
from os.path import join
import torch.distributed as dist


class Trainer():
    
    def __init__(self, criterion = None,
                 optimizer = None,
                 device = None,
                 scaler=None,
                 start_epoch=0,
                 lr_scheduler=None,
                 wd_scheduler=None,
                 augmentator=None,
                 save_path=None,
                 log_step=10,
                 is_distributed=True):
        
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.start_epoch = start_epoch
        self.lr_scheduler = lr_scheduler
        self.wd_scheduler = wd_scheduler
        self.augmentator = augmentator
        
        self.log_step = log_step
        self.is_distributed = is_distributed
        
        
        if save_path is not None and not os.path.exists(save_path):
            os.mkdir(save_path)
        
        self.save_path = save_path
        self.scaler = scaler

        
        
    def accuracy(self, logits, targets):
        ps = torch.argmax(logits,dim = 1).detach().cpu().numpy()
        acc = accuracy_score(ps,targets.detach().cpu().numpy())
        return acc
    
        
    def train_epoch(self, model, train_loader):
        epoch_loss = 0.0
        epoch_acc = 0.0
        batch_num = len(train_loader)
        for it, data in enumerate(train_loader):
            
            images, targets = data
            # images = images.to(self.device)
            # targets = targets.to(self.device)
            
            with torch.cuda.amp.autocast():
                global_feat, local_feat, logits = model(images, targets)
                loss = self.criterion(global_feat, local_feat, logits, targets)
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            epoch_loss += loss.item()
            epoch_acc += self.accuracy(logits, targets)
            
            log_content = {'B#': {it+1}/{batch_num},
                           'loss' : round(float(epoch_loss/(it+1)), 4),
                           'Acc' : round(float(epoch_acc/(it+1)), 4)}
            
            if (it + 1) % self.log_ste == 0:
                    self.log(log_content)
            

        return epoch_loss / len(train_loader), epoch_acc / len(train_loader)
    
    def valid_epoch(self, model, valid_loader):
        epoch_loss = 0.0
        epoch_acc = 0.0
        batch_num = len(valid_loader)
        for it, data in enumerate(valid_loader):
            
            images, targets = data
            # images = images.to(self.device)
            # targets = targets.to(self.device)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    global_feat, local_feat, logits = model(images, targets)
                    loss = self.criterion(global_feat, local_feat, logits, targets)
            
            epoch_loss += loss.item()
            epoch_acc += self.accuracy(logits, targets)
            
            log_content = {'B#': {it+1}/{batch_num},
                           'loss' : round(float(epoch_loss/(it+1)), 4),
                           'Acc' : round(float(epoch_acc/(it+1)), 4)}
            
            if (it + 1) % self.log_ste == 0:
                    self.log(log_content, is_train=False)
            
        return epoch_loss / len(valid_loader), epoch_acc / len(valid_loader)


    def log(self, content, is_train=True, only_main_device=False):
        if only_main_device and self.gpu != 0:
            return
        
        if not only_main_device:
            content['device'] = self.device
            
        log_str = '=>' + ' '.join("| {}: {}".format(k, v) for k, v in content.items())
        
        if self.is_distributed and self.device != 0: return
        
        with open(join(self.save_path, 'train_log.txt' if is_train else 'val_log.txt'), 'a+') as f:
            f.write(log_str)
    
    
    def run(self, model, train_loader, valid_loader=None, epochs_num=1):
        
        for epoch in range(self.start_epoch+1, epochs_num+1, 1):
            
            if self.lr_scheduler is not None:
                for g in self.optimizer.param_groups:
                    g['lr'] = self.lr_scheduler(epoch)
                
                    
            if self.wd_sceduler is not None:
                for g in self.optimizer.param_groups:
                    g['weight_decay'] = self.wd_scheduler(epoch-1)
            
            if self.augmentator is None:
                self.augmentator.update_augmentation_list(epoch-1)
            
            model.train()
            self.log({'Epoch ': epoch}, only_main_device=True)
            avg_train_loss, avg_train_acc = self.train_epoch(model, train_loader)
            
            if self.is_distributed: self.cleanup_distributed()
                
            if self.save_path is not None and self.is_distributed and self.device == 0:
                torch.save(model.module.state_dict(), join(self.save_path, f'epoch_{epoch}_model.pt'))
            
            if valid_loader is not None:
                model.eval()
                avg_valid_loss, avg_valid_acc = self.valid_epoch(model, valid_loader)
            
        return model
    
    def cleanup_distributed(self):
        dist.destroy_process_group()