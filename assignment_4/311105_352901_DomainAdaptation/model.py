import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 10, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(10)
        self.conv2 = nn.Conv1d(10, 10, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(10)
        self.conv3 = nn.Conv1d(10, 10, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(10)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(10 * 512, 256)  # 512 is input dimension

    def forward(self, x):
        x = self.dropout(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.classifier = Classifier()

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)