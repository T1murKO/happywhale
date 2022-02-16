from trainer import Trainer
from configs import Config
from utils import MaskImageDataset, get_augmentation_list
import pandas as pd
from torch.utils.data import DataLoader

config = Config()

data_csv = pd.read_csv(config.csv_path)
img_data = config.images_path

train_dataset = MaskImageDataset(data_csv,
                             img_data,
                             transform=config.train_transforms)

train_loader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle=True)

trainer = Trainer(criterion=config.loss,
                  optimizer=config.optimizer,
                  device=config.device,
                  start_epoch=config.start_epoch)

trainer.run(config.model, train_loader, epochs=config.num_epoch, save_path=config.save_path, schedule=config.schedule)

