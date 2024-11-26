import torch
import torch.nn as nn
from torchvision import models

class ResNetCheXpert(nn.Module):
    def __init__(self, resnet_type="resnet18", num_classes=5, pretrained=True):
        """
        Wrapper class for ResNet to adapt it for the CheXpert dataset.
        Args:
            resnet_type (str): Type of ResNet to use ("resnet18", "resnet34", "resnet50", etc.).
            num_classes (int): Number of output classes for the dataset.
            pretrained (bool): Whether to use a model pre-trained on ImageNet.
        """
        super(ResNetCheXpert, self).__init__()
        
        # Load the pre-trained ResNet model
        if resnet_type == "resnet18":
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif resnet_type == "resnet34":
            self.model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        elif resnet_type == "resnet50":
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif resnet_type == "resnet101":
            self.model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        elif resnet_type == "resnet152":
            self.model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        else:
            raise ValueError("Unsupported ResNet type. Choose from 'resnet18', 'resnet34', 'resnet50', 'resnet101', or 'resnet152'.")

        self.num_classes = num_classes
        # Modify the final fully connected layer to match the number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        # Optional: Add a sigmoid activation for multi-label classification
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = self.model(x)
        # x = self.sigmoid(x)  # Apply sigmoid activation for multi-label classification
        return x