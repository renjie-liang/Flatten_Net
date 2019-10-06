#test mul_out_channels backward
# ctx.needs_input_grad
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim

from Flatten_conv1d import flatten_conv2d 



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

criterion = nn.MSELoss()
batch_size = 2
in_channels = 3
out_channels = 6
row = 10
col = 13
kernel_size = 5
torch.manual_seed(0)
my_input = torch.rand([batch_size,in_channels,row,col])
my_input = my_input.to(device)



S = flatten_conv2d(in_channels= in_channels, 
                   out_channels= out_channels , 
                   kernel_size=kernel_size)
S.to(device)

my_output = S(my_input )
label = torch.ones_like(my_output)


loss = criterion(my_output, label)

loss.backward()



conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)


conv2.to(device)
my_output2 = conv2(my_input)

print('flatten',my_output.size())
print('conv2d',my_output2.size())