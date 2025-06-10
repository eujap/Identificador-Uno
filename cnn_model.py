import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # Imagens com 3 canais (RGB)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # 64 filtros, imagem final 16x16 após pooling
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # [B, 32, 32, 32]
        x = self.pool(torch.relu(self.conv2(x)))  # [B, 64, 16, 16]
        x = x.view(x.size(0), -1)                 # Flatten para [B, 64*16*16]
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
