import torch.nn as nn
import torchvision.models as models
import torch

class DiagnosticModel(nn.Module):

    def __init__(self, model_name = 'resnet18', in_channels = 3):
        super().__init__()
        
        num_classes = 2

        # Download ResNet-18 & Update Final Layer
        if model_name == 'resnet18':
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif model_name == 'resnet34':
            self.model = models.resnet34(weights =models.ResNet34_Weights.IMAGENET1K_V1)
        elif model_name == 'resnet50':
            self.model = models.resnet50(weights =models.ResNet50_Weights.IMAGENET1K_V1)
        elif model_name == 'convnext_tiny':
            self.model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        
        # Update First Layer for 2-channel input
        if in_channels == 2:
            if 'resnet' in model_name:
                # Save pretrained weights
                pretrained_w = self.model.conv1.weight  # (64, 3, 7, 7)

                # Replace conv1 to accept in_channels (2)
                self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

                # Initialize new conv weights
                with torch.no_grad():
                    self.model.conv1.weight[:, 0:1, :, :] = pretrained_w[:, 0:1, :, :]       # actual image
                    self.model.conv1.weight[:, 1:2, :, :] = torch.randn(64, 1, 7, 7) * 0.01  # mask channel


        # Update Final Layer for Binary Classification
        if 'resnet' in model_name:
            self.model.fc = nn.Linear(self.model.fc.in_features, 2) # binary classification
        elif 'convnext' in model_name:
            self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, num_classes)

    def forward(self, x):
        return self.model(x)




