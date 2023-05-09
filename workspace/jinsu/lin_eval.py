import torch
import torch.nn as nn

from resnet import resnet50
model_dict = {'resnet50': [resnet50, 2048]}

class LinearClassifier(nn.Module):
    def __init__(self, name='resnet50', num_classes=2):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)