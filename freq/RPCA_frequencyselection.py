import torch
import torch.nn as nn
import torch.nn.init as init
import copy
import math

def replace_with_parallel_transformer(model, target_module_name):
    """
    Replace the target Transformer2DModel with a parallel version

    Args:
        model: The entire pre-trained model
        target_module_name: Full path name of the target module to replace
    """
    # Locate the target module
    module_names = target_module_name.split('.')
    parent_module = model
    for name in module_names[:-1]:
        parent_module = getattr(parent_module, name)

    original_module = getattr(parent_module, module_names[-1])
    downsam = None

    parallel_module = ReVi_inout(original_module, downsam)

    # Replace the module
    setattr(parent_module, module_names[-1], parallel_module)

    return model

class ReVi_inout(nn.Module):
    def __init__(self, stave_inout, downsam):
        super().__init__()
        # Create the original module
        self.downsam = downsam
        self.stave_inout = copy.deepcopy(stave_inout)
        channel = self.stave_inout.dim
        down_channel = 64
        self.dconv0 = nn.Sequential(*[nn.Conv2d(channel, down_channel, kernel_size=1, padding=0, stride=1),
                         #nn.BatchNorm2d(down_channel),
                         nn.ReLU(True)])
        self.upconv = nn.Sequential(*[nn.Conv2d(down_channel, channel, kernel_size=1, padding=0, stride=1),
                         #nn.BatchNorm2d(down_channel),
                         nn.ReLU(True)])
        self.SKA_convs = nn.Sequential(*[nn.Conv2d(down_channel, down_channel, kernel_size=3, padding=1, stride=1),
                         nn.BatchNorm2d(down_channel),
                         nn.ReLU(True)])

        self.MKE_convs = nn.Sequential(*[nn.Conv2d(down_channel, down_channel, kernel_size=3, padding=1, stride=1),
                         nn.BatchNorm2d(down_channel),
                         nn.ReLU(True),
                         nn.Conv2d(down_channel, down_channel, kernel_size=3, padding=1, stride=1)])

        self.selmod = nn.Sequential(*[nn.Conv2d(down_channel, down_channel, kernel_size=3, padding=1, stride=1),
                                         nn.BatchNorm2d(down_channel),
                                         nn.ReLU(True)])
        self._initialize_weights()

        for param in self.SKA_convs.parameters():
            param.requires_grad = True
        for param in self.MKE_convs.parameters():
            param.requires_grad = True
        for param in self.selmod.parameters():
            param.requires_grad = True
        for param in self.dconv0.parameters():
            param.requires_grad = True
        for param in self.upconv.parameters():
            param.requires_grad = True
        self.epsilon = nn.Parameter(torch.Tensor([0.01]), requires_grad=True)
        self.alpha = nn.Parameter(torch.Tensor([0.01]), requires_grad=True)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(
            self,
            x: torch.Tensor,
            o,
            mask = None,
    ):
        # Forward pass of the original branch
        original_output, wst = self.stave_inout(x, o)

        inner_dim = x.shape[2]
        o_dim = 64
        batch = x.shape[0]
        height = math.isqrt(x.shape[1])
        width = height
        if height * height != x.shape[1]:
            print("no")
        x_2ud = x.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2)
        o_2d = o.reshape(batch, height, width, o_dim).permute(0, 3, 1, 2)
        x_2d = self.dconv0(x_2ud)
        b0 = x_2d - o_2d
        bk = self.SKA_convs(b0) + b0
        xo = x_2d + o_2d - bk

        dct_output = self.selmod(xo)
        w = dct_output + xo
        
        ok_2d = xo - self.MKE_convs(w) * self.epsilon
        ok_2ud = self.upconv(ok_2d)
        ok = ok_2ud.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)

        output = original_output + ok * self.alpha #- bk * self.beta
        ok = ok_2d.permute(0, 2, 3, 1).reshape(batch, height * width, o_dim)

        return output, ok


