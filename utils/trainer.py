from sklearn.metrics import accuracy_score
import torch
from tqdm import tqdm
import os
from os.path import join


class Trainer():
    
    def __init__(self, criterion = None, optimizer = None, device = None, start_epoch=0, mixed_presicion=False):
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.start_epoch = start_epoch

        self.train_batch = self.train_batch_loop
        if mixed_presicion:
            self.scaler = torch.cuda.amp.GradScaler()
            self.train_batch = self.train_batch_loop_mixed
        
        
    def accuracy(self, logits, targets):
        ps = torch.argmax(logits,dim = 1).detach().cpu().numpy()
        acc = accuracy_score(ps,targets.detach().cpu().numpy())
        return acc
        
    def train_batch_loop_mixed(self, model, train_loader, i, save_path=None, log_path=None):
        epoch_loss = 0.0
        epoch_acc = 0.0
        pbar_train = tqdm(train_loader, desc="Epoch" + " [TRAIN] " + str(i+1))
        batch_num = len(pbar_train)
        for it, data in enumerate(pbar_train):
            
            images, targets = data
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            with torch.cuda.amp.autocast():
                logits = model(images, targets)
                loss = self.criterion(logits,targets)
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            epoch_loss += loss.item()
            epoch_acc += self.accuracy(logits, targets)
            
            postfix = {'loss' : round(float(epoch_loss/(it+1)), 4), 'acc' : float(epoch_acc/(it+1))}
            pbar_train.set_postfix(postfix)
            
            if save_path is not None:
                if it % 100 == 99:
                    with open(join(log_path, 'train_log.txt'), 'a') as f:
                        f.write(f'B# {it+1}/{batch_num}, Loss: {round(float(epoch_loss/(it+1)), 4)}, Acc: {round(float(epoch_acc/(it+1)), 4)} \n')
                
            
        return epoch_loss / len(train_loader), epoch_acc / len(train_loader)

    
    def train_batch_loop(self, model, train_loader, i, save_path=None, log_path=None):
        epoch_loss = 0.0
        epoch_acc = 0.0
        pbar_train = tqdm(train_loader, desc="Epoch" + " [TRAIN] " + str(i+1))
        batch_num = len(pbar_train)
        for it, data in enumerate(pbar_train):
            
            images, targets = data
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            logits = model(images, targets)
            loss = self.criterion(logits,targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += self.accuracy(logits, targets)
            
            postfix = {'loss' : round(float(epoch_loss/(it+1)), 4), 'acc' : float(epoch_acc/(it+1))}
            pbar_train.set_postfix(postfix)
            
            if save_path is not None:
                if it % 100 == 99:
                    with open(join(log_path, 'train_log.txt'), 'a') as f:
                        f.write(f'B# {it+1}/{batch_num}, Loss: {round(float(epoch_loss/(it+1)), 4)}, Acc: {round(float(epoch_acc/(it+1)), 4)} \n')
                
            
        return epoch_loss / len(train_loader), epoch_acc / len(train_loader)
            
    
    def valid_batch_loop(self, model, valid_loader, i, save_path=None):
        
        epoch_loss = 0.0
        epoch_acc = 0.0
        pbar_valid = tqdm(valid_loader, desc = "Epoch" + " [VALID] " + str(i+1))
        batch_num = len(pbar_valid)
        
        for it, data in enumerate(pbar_valid):
            
            images,targets = data
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            logits = model(images, targets)
            loss = self.criterion(logits, targets)
            
            epoch_loss += loss.item()
            epoch_acc += self.accuracy(logits, targets)
            
            postfix = {'loss' : round(float(epoch_loss/(it+1)), 4), 'acc' : float(epoch_acc/(it+1))}
            pbar_valid.set_postfix(postfix)
            
            
            if save_path is not None:
                if it % 200 == 199:
                    with open(join(save_path, 'valid_log.txt'), 'a') as f:
                        f.write(f'B# {it+1}/{batch_num}, Loss: {round(float(epoch_loss/(it+1)), 4)}, Acc: {round(float(epoch_acc/(it+1)), 4)} \n')
            
        return epoch_loss / len(valid_loader), epoch_acc / len(valid_loader)
            
    
    def run(self, model, train_loader, valid_loader=None, schedule=None, epochs=1, save_path=None, mixed_presicion=False):
        if not os.path.exists(save_path) and save_path is not None:
            os.mkdir(save_path)
        
        for i in range(self.start_epoch, epochs, 1):
            if save_path is not None:
                
                with open(join(save_path, 'train_log.txt'), 'a') as f:
                        f.write(f'---- EPOCH {i} ----\n')
                
                epoch_save_path = join(save_path, f'epoch_{i}/')
                if not os.path.exists(epoch_save_path):
                    os.mkdir(epoch_save_path)
            else:
                epoch_save_path = None
            
            if schedule is not None:
                for g in self.optimizer.param_groups:
                    g['lr'] = schedule(i)
            
            model.train()
            avg_train_loss, avg_train_acc = self.train_batch(model, train_loader, i, save_path=epoch_save_path, log_path=save_path)
            
            if save_path is not None:
                torch.save(model.module.state_dict(), join(epoch_save_path, 'model.pth'))
            
            if valid_loader is not None:
                model.eval()
                avg_valid_loss, avg_valid_acc = self.valid_batch_loop(model, valid_loader, i, save_path=epoch_save_path)
            
        return model
    
    def run_eval(self, model, data_lodaer):
        model.eval()
        avg_valid_loss, avg_valid_acc = self.valid_batch_loop(model, data_lodaer, 0)
        return avg_valid_loss, avg_valid_acc