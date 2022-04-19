import torch
from torch import nn


class Model_CNN(nn.Module):
    def __init__(self, n_in, n_out, nh=16):
        super(Model_CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # halves img dim
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=nh,
                out_channels=2*nh,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # halves img dim
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=2*nh,
                out_channels=2*2*nh,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # halves img dim
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(14400, n_out)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        out = self.fc(x)
        return out


class Model_1(nn.Module):
    def __init__(self, n_in, n_out, nh=8):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_in, nh),
            nn.ReLU(),
            nn.Linear(nh, n_out)
            # nn.Linear(n_in, 128),
            # nn.ReLU(),
            # # nn.Dropout(p=0.2),

            # nn.Linear(128, 64),
            # nn.ReLU(),
            # # nn.Dropout(p=0.2),

            # nn.Linear(64, 32),
            # nn.ReLU(),
            # # nn.Dropout(p=0.2),

            # nn.Linear(32, n_out)
        )

    def forward(self, x):
        return self.model(x)


class Model_2(nn.Module):
    def __init__(self, n_in, n_out, nh=512):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_in, 8192),
            nn.ReLU(),
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(32, n_out)
        )

    def forward(self, x):
        return self.model(x)


def get_model(data, arch, lr, c, opt, device):
    input_shape = data.train_ds.x.shape[1]
    net = globals()[arch](input_shape, c).to(device)
    optim = getattr(torch.optim, opt)
    return net, optim(net.parameters(), lr=lr)
