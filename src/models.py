import numpy as np
import torch
import torch.nn as nn

class EfficientNet():
    pass

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        resnet = torch.hub.load('pytorch/vision:v0.10.0',
                                'resnet18', pretrained=True)
        num_ftrs = resnet.fc.in_features
        self.model = nn.Sequential(*(list(resnet.children())[:-1]))
        set_parameter_requires_grad(self.model, False)
        self.digit1 = nn.Linear(num_ftrs, num_classes)
        self.digit2 = nn.Linear(num_ftrs, num_classes)
        self.digit3 = nn.Linear(num_ftrs, num_classes)
        self.digit4 = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        x1 = self.digit1(x)
        x2 = self.digit2(x)
        x3 = self.digit3(x)
        x4 = self.digit4(x)
        return torch.stack([x1,x2,x3,x4], dim=2)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False