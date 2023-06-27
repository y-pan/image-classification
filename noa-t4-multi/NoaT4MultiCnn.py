import torch
import torch.nn as nn

import vars

class NoaT4MultiCnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5,5)),
            nn.ReLU(),
            nn.MaxPool2d((4,4))
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5,5)),
            nn.ReLU(),
            nn.MaxPool2d((4,4))
        )

        self.fc = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(4624, 1156),
            nn.ReLU(),
            nn.Linear(1156, vars.NUM_CLASSES) # produces logits
        )

    def forward(self, x):
        assert x.size()[1:] == torch.Size([vars.IMAGE_CHANNELS, vars.IMAGE_HEIGHT, vars.IMAGE_WIDTH]), \
            f"Expecting image shape: torch.Size([{vars.IMAGE_CHANNELS}, {vars.IMAGE_HEIGHT}, {vars.IMAGE_WIDTH}]), while {x.size()[1:]} provided"
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x
        
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.randn((8, vars.IMAGE_CHANNELS, vars.IMAGE_HEIGHT, vars.IMAGE_WIDTH), dtype=torch.float, device=device)
    model = NoaT4MultiCnn().to(device=device)
    pred = model(X)
    print("It didn't crash at least!")