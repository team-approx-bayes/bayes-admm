# Models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# A linear model
class LinearModel(torch.nn.Module):
    def __init__(self, D_in, D_out, bias=False):
        super(LinearModel, self).__init__()
        # Only one linear layer
        self.upper = nn.Linear(D_in, D_out, bias=bias)

        # Initialise parameter values to zeros
        torch.nn.init.zeros_(self.upper.weight)

    # Forward pass through model
    def forward(self, x):
        h_act = x
        y_pred = self.upper(h_act)
        return y_pred

# Define a deterministic MLP with hidden layer size: hidden_sizes[0], ..., hidden_sizes[-1]
class MLP(torch.nn.Module):
    def __init__(self, D_in, hidden_sizes, D_out, act_func="relu"):
        super(MLP, self).__init__()

        # Hidden layers
        self.linear = torch.nn.ModuleList()
        weight_matrix = [D_in] + hidden_sizes
        for i in range(len(hidden_sizes)):
            self.linear.append(nn.Linear(weight_matrix[i], weight_matrix[i+1]))

        # Upper layer
        self.upper = nn.Linear(hidden_sizes[-1], D_out)

        # Set activation function
        if act_func == "relu":
            self.act_function = torch.nn.ReLU()
        elif act_func == "sigmoid":
            self.act_function = torch.nn.Sigmoid()
        elif act_func == "tanh":
            self.act_function = torch.nn.Tanh()
        else:
            raise ValueError("Cannot yet implement activation %s" % act_func)


    # If output_range is not None, then only output some classes' values (cf a multi-head setup)
    def forward(self, x, output_range=None):

        x = x.squeeze()
        if len(x.shape) == 1:
            x = torch.unsqueeze(x,0)
        h_act = x
        for i in range(len(self.linear)):
            h_act = self.linear[i](h_act)
            h_act = self.act_function(h_act)

        y_pred = self.upper(h_act)

        # Only output some heads (cf a multi-head setup in continual learning)
        if output_range is not None:
            y_pred = y_pred[:, output_range]

        return y_pred

# Define a CNN for CIFAR datasets
class CifarNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(type(self), self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.25),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.25)
        )

        self.linear_block = nn.Sequential(
            nn.Linear(64*6*6, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.upper = nn.Linear(512, out_channels)
        #self.weight_init()

    def weight_init(self):
        nn.init.constant_(self.upper.weight, 0)
        nn.init.constant_(self.upper.bias, 0)

    def forward(self, x, output_range=None):
        o = self.conv_block(x)
        o = torch.flatten(o, 1)
        o = self.linear_block(o)
        o = self.upper(o)

        # Only output some heads (cf a multi-head setup in continual learning)
        if output_range is not None:
            o = o[:, output_range]
        return o

# Define a CNN for CIFAR datasets (the architecture is from the FedDyn paper)
class FedDynCifarCNN(nn.Module):
    def __init__(self, n_cls = 10):
        super(type(self), self).__init__()
        
        self.n_cls = n_cls 
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64*5*5, 384) 
        self.fc2 = nn.Linear(384, 192) 
        self.fc3 = nn.Linear(192, self.n_cls)

        # self.init_weights()

    def forward(self, x, output_range=None):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)



# The following code for FilterResponseNorm is from https://github.com/gupta-abhay/pytorch-frn
class FilterResponseNormNd(nn.Module):

    def __init__(self, ndim, num_features, eps=1e-6,
                 learnable_eps=False):
        """
        Input Variables:
        ----------------
            ndim: An integer indicating the number of dimensions of the expected input tensor.
            num_features: An integer indicating the number of input feature dimensions.
            eps: A scalar constant or learnable variable.
            learnable_eps: A bool value indicating whether the eps is learnable.
        """
        assert ndim in [3, 4, 5], \
            'FilterResponseNorm only supports 3d, 4d or 5d inputs.'
        super(FilterResponseNormNd, self).__init__()
        shape = (1, num_features) + (1,) * (ndim - 2)
        self.eps = nn.Parameter(torch.ones(*shape) * eps)
        if not learnable_eps:
            self.eps.requires_grad_(False)
        self.gamma = nn.Parameter(torch.Tensor(*shape))
        self.beta = nn.Parameter(torch.Tensor(*shape))
        self.tau = nn.Parameter(torch.Tensor(*shape))
        self.reset_parameters()

    def forward(self, x):
        avg_dims = tuple(range(2, x.dim()))
        nu2 = torch.pow(x, 2).mean(dim=avg_dims, keepdim=True)
        x = x * torch.rsqrt(nu2 + torch.abs(self.eps))
        return torch.max(self.gamma * x + self.beta, self.tau)

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        nn.init.zeros_(self.tau)


class FilterResponseNorm1d(FilterResponseNormNd):

    def __init__(self, num_features, eps=1e-6, learnable_eps=False):
        super(FilterResponseNorm1d, self).__init__(
            3, num_features, eps=eps, learnable_eps=learnable_eps)


class FilterResponseNorm2d(FilterResponseNormNd):

    def __init__(self, num_features, eps=1e-6, learnable_eps=False):
        super(FilterResponseNorm2d, self).__init__(
            4, num_features, eps=eps, learnable_eps=learnable_eps)


class FilterResponseNorm3d(FilterResponseNormNd):

    def __init__(self, num_features, eps=1e-6, learnable_eps=False):
        super(FilterResponseNorm3d, self).__init__(
            5, num_features, eps=eps, learnable_eps=learnable_eps)

# ResNet20 architecture code
# Code is from https://github.com/akamaster/pytorch_resnet_cifar10
def _weights_init(m):
    #classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', normalisation='GroupNorm'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        if normalisation == 'GroupNorm':
            self.bn1 = nn.GroupNorm(2, planes)
        else:
            self.bn1 = FilterResponseNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if normalisation == 'GroupNorm':
            self.bn2 = nn.GroupNorm(2, planes)
        else:
            self.bn2 = FilterResponseNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, normalisation='GroupNorm', num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.normalisation = normalisation

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        if self.normalisation == 'GroupNorm':
            self.bn1 = nn.GroupNorm(2, 16)
        else:
            self.bn1 = FilterResponseNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, normalisation=self.normalisation)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, normalisation=self.normalisation)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, normalisation=self.normalisation)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, normalisation):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, normalisation=normalisation))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if self.normalisation == 'GroupNorm':
            out = nn.AdaptiveAvgPool2d((1, 1))(out)
        else:
            out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet20(num_classes=10, normalisation='GroupNorm'):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, normalisation=normalisation)

