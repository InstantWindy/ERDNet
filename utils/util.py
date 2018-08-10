import torch
import torch.nn as nn
import PIL.Image
from io import BytesIO
from IPython.display import clear_output, Image, display
import numpy as np

lr_pow=0.9
learning_rate=1e-2

class PolyDecay:
    def __init__(self, initial_lr, power, n_epochs):
        self.initial_lr = initial_lr
        self.power = power
        self.n_epochs = n_epochs
    
    def scheduler(self, epoch,optimizer):
        lr=self.initial_lr * np.power(1.0 - 1.0*epoch/self.n_epochs, self.power)
        optimizer.param_groups[0]['lr']=lr
#         return self.initial_lr * np.power(1.0 - 1.0*epoch/self.n_epochs, self.power)
            
class ExpDecay:
    def __init__(self, initial_lr, decay):
        self.initial_lr = initial_lr
        self.decay = decay
    
    def scheduler(self, epoch,optimizer):
        lr=self.initial_lr * np.exp(-self.decay*epoch)
        optimizer.param_groups[0]['lr']=lr
        
#         return self.initial_lr * np.exp(-self.decay*epoch)




# 定义 bilinear kernel
def bilinear_kernel(in_channels, out_channels, kernel_size):
    '''
    return a bilinear filter tensor
    '''
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num, group_num = 16, eps = 1e-10):
        super(GroupBatchnorm2d,self).__init__()
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.ones(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()

        x = x.view(N, self.group_num, -1)

        mean = x.mean(dim = 2, keepdim = True)
        std = x.std(dim = 2, keepdim = True)

        x = (x - mean) / (std+self.eps)
        x = x.view(N, C, H, W)

        return x * self.gamma + self.beta
    
def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
    """Sets the learning rate to the initially
        configured `lr` decayed by `decay` every `n_epochs`"""
    new_lr = lr * (decay ** (cur_epoch / n_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    
    
# def lr_poly(base_lr, iter, max_iter, power):
#     return base_lr * ((1 - float(iter) / max_iter) ** (power))

	
# def adjust_learning_rate(optimizer, i_iter,max_iter):
#     lr = lr_poly(learning_rate, i_iter, max_iter, lr_pow)
#     optimizer.param_groups[0]['lr'] = lr
#     if len(optimizer.param_groups) > 1:
#         optimizer.param_groups[1]['lr'] = lr * 10