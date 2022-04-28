import torch
from torch import nn


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, *args):
        super(InceptionModule, self).__init__()
        relu = nn.ReLU()
        self.branch1 = nn.Sequential(
                  nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
                  relu)

        conv3_1 = nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        conv3_3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.branch2 = nn.Sequential(conv3_1, conv3_3,relu)

        conv5_1 = nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        conv5_5 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.branch3 = nn.Sequential(conv5_1,conv5_5,relu)

        max_pool_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        conv_max_1 = nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.branch4 = nn.Sequential(max_pool_1, conv_max_1,relu)

    def forward(self, input):
        output1 = self.branch1(input)
        output2 = self.branch2(input)
        output3 = self.branch3(input)
        output4 = self.branch4(input)
        out = torch.cat([output1, output2, output3, output4], dim=1)
        print(out.shape)
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

def get_model(data, arch, lr, opt, device):
    data_shape = data.train_dl.dataset[0][0].shape
    n_in = data_shape[0]
    nh = arch[2]
    n_out = data.c
    
    arch_model = arch[0]
    arch_depth = arch[1]
    
    img_shape = data_shape[1]
    optim = getattr(torch.optim, opt)
    
    net = globals()[arch_model](n_in, n_out, nh, img_shape, arch_depth).to(device)
    return net, optim(net.parameters(), lr=lr)

class AlexNet(nn.Module):
    def __init__(self, n_in, n_out, nh, img_size, num_blocks, kernel_size=3, stride=1, padding=1) -> None:
        super(AlexNet, self).__init__()
        self. n_in = n_in
        self.c = n_out

        self.features = nn.Sequential(
            nn.Conv2d(n_in, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )   
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.c),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
