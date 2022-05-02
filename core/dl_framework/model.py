import torch
from torch import nn

#always format: (b, c, h, w) "channel first"
#output of layer:  floor[(input + 2*padding — kernel) / stride + 1]
class ResBlock(nn.Module):
    def __init__(self, n_in, n_out, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(n_in, n_out, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(n_in, n_out, kernel_size=1, stride=2),
                nn.BatchNorm2d(n_out)
            )
        else:
            self.conv1 = nn.Conv2d(n_in, n_out, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()
        
        self.conv2 = nn.Conv2d(n_out, n_out, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.bn2 = nn.BatchNorm2d(n_out)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = nn.ReLU()(self.bn2(self.conv2(x)))
        out = nn.ReLU()(x + shortcut)
        return out

class ResNet18(nn.Module):
    def __init__(self, n_in, n_out, *args):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(n_in, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = nn.Sequential(
            ResBlock(64, 64, downsample=False),
            ResBlock(64, 64, downsample=False)
        )
        self.layer2 = nn.Sequential(
            ResBlock(64, 128, downsample=True),
            ResBlock(128, 128, downsample=False)
        )
        self.layer3 = nn.Sequential(
            ResBlock(128, 256, downsample=True),
            ResBlock(256, 256, downsample=False)
        )
        self.layer4 = nn.Sequential(
            ResBlock(256, 512, downsample=True),
            ResBlock(512, 512, downsample=False)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, n_out)
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        out = self.fc(x)
        return out

class ResNet34(nn.Module):
    def __init__(self, n_in, n_out, *args):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(n_in, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = nn.Sequential(
            ResBlock(64, 64, downsample=False),
            ResBlock(64, 64, downsample=False),
            ResBlock(64, 64, downsample=False),
        )
        self.layer2 = nn.Sequential(
            ResBlock(64, 128, downsample=True),
            ResBlock(128, 128, downsample=False),
            ResBlock(128, 128, downsample=False),
            ResBlock(128, 128, downsample=False),
        )
        self.layer3 = nn.Sequential(
            ResBlock(128, 256, downsample=True),
            ResBlock(256, 256, downsample=False),
            ResBlock(256, 256, downsample=False),
            ResBlock(256, 256, downsample=False),
            ResBlock(256, 256, downsample=False),
            ResBlock(256, 256, downsample=False),
        )
        self.layer4 = nn.Sequential(
            ResBlock(256, 512, downsample=True),
            ResBlock(512, 512, downsample=False),
            ResBlock(512, 512, downsample=False),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, n_out)
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        out = self.fc(x)
        return out

class CustomModel(nn.Module):
    def __init__(self, n_in, n_out, nh, img_size, num_blocks, kernel_size=3, stride=1, padding=1):    
        super(CustomModel, self).__init__()
        modules_list = []
        for n in range(num_blocks):
            modules, output_size = conv_block(img_size, n_in, nh, kernel_size, stride, padding)
            modules_list.extend(modules)
            
            img_size = output_size
            n_in = nh
            nh = 2*nh
        self.conv = nn.Sequential(*modules_list)
        output_size = output_size**2 * n_in

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(output_size, n_out)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        out = self.fc(x)
        return out

def conv_output_size(w, k=3, p=2, s=1):
    return ((w - k + 2*p) / s + 1)

def conv_block(img_size, n_in, nh, kernel_size, stride, padding, max_kernel=2):
    modules = [
    nn.Conv2d(n_in, nh, kernel_size, stride, padding),
    nn.BatchNorm2d(nh),
    nn.ReLU(),
    nn.MaxPool2d(max_kernel)
    ]
    output_size = conv_output_size(img_size, k=kernel_size, p=padding, s=stride)
    output_size = conv_output_size(output_size, k=max_kernel, p=0, s=max_kernel)
    return modules, int(output_size)

def get_model(data, arch, lr, opt, device, ext_model=None):
    # TODO: add variable for option to train model with fixed layers
    data_shape = data.train_dl.dataset[0][0].shape
    n_in = data_shape[0]
    nh = arch[2]
    n_out = data.c
    
    arch_model = arch[0]
    arch_depth = arch[1]
    
    img_shape = data_shape[1]
    optim = getattr(torch.optim, opt)
    if not ext_model:
        net = globals()[arch_model](n_in, n_out, nh, img_shape, arch_depth).to(device)
    else:
        # try:
        #     num_ftrs = ext_model.fc.in_features
        #     ext_model.fc = nn.Linear(num_ftrs, n_out)
        #     optimizer = optim(net.parameters(), lr=lr)
        # except: 
        num_ftrs = ext_model.classifier[-1].in_features
        ext_model.classifier[-1] = nn.Linear(num_ftrs, n_out)
        for param in ext_model.features.parameters():
            param.requires_grad = False
        optimizer = optim(filter(lambda p: p.requires_grad, ext_model.parameters()), lr=lr)
        net = ext_model.to(device)
    return net, optimizer

