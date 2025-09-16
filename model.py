import torch
import torch.nn as nn
import torchvision.models as models


class DiagnosticModel(nn.Module):

    def __init__(self):
        super().__init__()

        # Download ResNet-18 & Update Final Layer
        self.res18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.res18.fc = nn.Linear(self.res18.fc.in_features, 2) # binary classification

    
    def forward(self, x):
        return self.res18(x)




