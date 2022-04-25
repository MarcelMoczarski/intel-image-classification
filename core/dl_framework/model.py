import torch
from torch import nn

3, 6, 16, 
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
    
    def forward(self, x):
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
