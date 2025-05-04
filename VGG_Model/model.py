import torch
import torch.nn as nn
import torch.ao.quantization as tq
import torch.nn.functional as F


class VGG(nn.Module):
    def __init__(self, in_channels=3, in_size=32, num_classes=10) -> None:
        super(VGG, self).__init__()
        
        # Conv1 (2 layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 32×32 -> 16×16
        )
        
        # Conv2 (2 layers)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 16×16 -> 8×8
        )
        
        # Conv3 (2 layers)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 8×8 -> 4×4
        )
        
        # Conv4 (Dilated, 1 layer)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Conv5 (1 layer)
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 4×4 -> 2×2
        )
        
        # Fully Connected Layers
        fmap_size = in_size // 16  # 32 -> 16 -> 8 -> 4 -> 2 (4 次 MaxPool)
        self.fc6 = nn.Sequential(
            nn.Linear(256 * fmap_size * fmap_size, 256),  # 256×2×2 = 1024
            nn.ReLU(inplace=True)
        )
        self.fc7 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True)
        )
        self.fc8 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)  # 32×32 -> 16×16
        x = self.conv2(x)  # 16×16 -> 8×8
        x = self.conv3(x)  # 8×8 -> 4×4
        x = self.conv4(x)  # 4×4 -> 4×4
        x = self.conv5(x)  # 4×4 -> 2×2
        x = torch.flatten(x, start_dim=1)  # Flatten: (batch_size, 1024)
        x = self.fc6(x)  # 1024 -> 256
        x = self.fc7(x)  # 256 -> 128
        x = self.fc8(x)  # 128 -> 10
        return x
        
    def fuse_modules(self):
        # fuse conv1 兩組
        self.eval()
        tq.fuse_modules(self.conv1, [['0', '1', '2'], ['3', '4', '5']], inplace=True)
        
        # fuse conv2
        tq.fuse_modules(self.conv2, [['0', '1', '2'], ['3', '4', '5']], inplace=True)

        # fuse conv3
        tq.fuse_modules(self.conv3, [['0', '1', '2'], ['3', '4', '5']], inplace=True)

        # fuse conv4 (只有一組)
        tq.fuse_modules(self.conv4, [['0', '1', '2']], inplace=True)

        # fuse conv5
        tq.fuse_modules(self.conv5, [['0', '1', '2']], inplace=True)

        # fuse 全連接層
        tq.fuse_modules(self.fc6, [['0', '1']], inplace=True)
        tq.fuse_modules(self.fc7, [['0', '1']], inplace=True)

        
            
if __name__ == "__main__":
    model = VGG()
    inputs = torch.randn(1, 3, 32, 32)
    print(model)

    from torchsummary import summary

    summary(model, (3, 32, 32), device="cpu")

