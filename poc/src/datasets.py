import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch
from torchvision import transforms

class CheXpertDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform
        self.image_paths = self.data["Path"].values
        self.labels = self.data.iloc[:, 5:].values
        self.label_names = self.data.columns[5:]
        
        print(f"Loaded {len(self.data)} samples from {csv_file}")
        print(f"Labels: {self.labels.shape}")
        print(f"Image paths: {self.image_paths.shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_paths[idx])
        image = Image.open(img_name).convert("RGB")
        labels = self.labels[idx]
        image = self.transform(image)
        labels = torch.tensor(labels, dtype=torch.int64)

        return image, labels
    
class CheXpertLitDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size=32):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        
    def __init__(self, data_path, batch_size=32):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size

        self.train_dataset = CheXpertDataset(
            csv_file=os.path.join(data_path, "train.csv"),
        )
        self.val_dataset = CheXpertDataset(
            csv_file=os.path.join(data_path, "valid.csv"),
        )
        self.test_dataset = None
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=6)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=6)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)