import torch
import torch.nn as nn

NUM_CLASSES = 2
IMAGE_CHANNELS = 1
IMAGE_HEIGHT = 300 
IMAGE_WIDTH = 300

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5,5)),
            nn.ReLU(),
            nn.MaxPool2d((4,4)),
            nn.Dropout2d(p=0.2)
        )
        # torch.Size([8, 16, 74, 74])
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5,5)),
            nn.ReLU(),
            nn.MaxPool2d((4,4)),
            nn.Dropout2d(p=0.2)
        )
        # torch.Size([8, 32, 17, 17])
        
        self.fc = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(32*17*17, 1156), # 9248 = 32*17*17
            nn.ReLU(),
            nn.Linear(1156, NUM_CLASSES) # produces logits
        )

    def forward(self, x):
        assert x.size()[1:] == torch.Size([IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH]), \
            f"Expecting image shape: torch.Size([{IMAGE_CHANNELS}, {IMAGE_HEIGHT}, {IMAGE_WIDTH}]), while {x.size()[1:]} provided"
        x = self.conv1(x)
        # print("conv1=>", x.size())
        x = self.conv2(x)
        # print("conv2=>",x.size())
        x = self.fc(x)
        # print("fc=>",x.size())
        return x
        
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.randn((8, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=torch.float, device=device)
    model = Model().to(device=device)
    pred = model(X)
    print("It didn't crash at least!")