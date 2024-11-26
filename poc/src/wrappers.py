import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import AUROC

from .models import ResNetCheXpert

def mask_nan(input_tensor):
    # mask = ~torch.isnan(input_tensor)
    # return input_tensor[mask]
    return torch.nan_to_num(input_tensor, nan=0.0)

class CheXpertLitModel(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.num_classes = 14
        self.model =  ResNetCheXpert(resnet_type="resnet50", num_classes=self.num_classes, pretrained=True)
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()
        self.metric = AUROC(task="multilabel", num_labels=self.num_classes, average="macro")
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(mask_nan(y_hat.float()), mask_nan(y.float()))
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(mask_nan(y_hat.float()), mask_nan(y.float()))
        
        self.metric.update(mask_nan(y_hat).sigmoid(), mask_nan(y).int())
        self.log("val_loss", loss, prog_bar=True)
        self.metric(y_hat, y)
        return loss
        
    def on_validation_epoch_end(self):
        val_auroc = self.metric.compute()
        self.log("val_auroc", val_auroc, prog_bar=True)
        self.metric.reset()
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        return [optimizer], [scheduler]