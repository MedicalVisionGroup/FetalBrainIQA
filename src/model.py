import torch.nn as nn
import torchvision.models as models


class DiagnosticModel(nn.Module):

    def __init__(self, model_name = 'resnet18'):
        super().__init__()

        # Download ResNet-18 & Update Final Layer
        if model_name == 'resnet18':
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif model_name == 'resnet34':
            self.model = models.resnet34(weights =models.ResNet34_Weights.IMAGENET1K_V1)
        elif model_name == 'resnet50':
            self.model = models.resnet50(weights =models.ResNet50_Weights.IMAGENET1K_V1)

        # Update Final Layer for Binary Classification
        self.model.fc = nn.Linear(self.model.fc.in_features, 2) # binary classification

    def forward(self, x):
        return self.model(x)




