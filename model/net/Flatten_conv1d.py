import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Function



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# test Module class
class flatten_conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, 
                kernel_size, stride=1, padding=0, 
                dilation=1, groups=1, bias=True, padding_mode='zeros'):

        super(flatten_conv2d, self).__init__()
        
#         self.in_row = in_row
#         self.in_col = in_col
        self.in_channels = in_channels
        self.out_channels = out_channels
#         self.kernel_size = kernel_size
        wight_size = self.in_channels
        self.L1_weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.L1_bias = nn.Parameter(torch.Tensor(out_channels))
        
        self.V1_weight = nn.Parameter(torch.Tensor(out_channels, kernel_size))
        self.V1_bias = nn.Parameter(torch.Tensor(out_channels))
        
        self.H1_weight = nn.Parameter(torch.Tensor(out_channels, kernel_size))
        self.H1_bias = nn.Parameter(torch.Tensor(out_channels))

        self.L2_weight = nn.Parameter(torch.Tensor(out_channels, out_channels))
        self.L2_bias = nn.Parameter(torch.Tensor(out_channels))
        
        self.V2_weight = nn.Parameter(torch.Tensor(out_channels, kernel_size))
        self.V2_bias = nn.Parameter(torch.Tensor(out_channels))
        
        self.H2_weight = nn.Parameter(torch.Tensor(out_channels, kernel_size))
        self.H2_bias = nn.Parameter(torch.Tensor(out_channels))




        self.L1_weight.data.uniform_(-0.1, 0.1)
        self.L1_bias.data.uniform_(-0.1, 0.1)
        
        self.V1_weight.data.uniform_(-0.1, 0.1)
        self.V1_bias.data.uniform_(-0.1, 0.1)
               
        self.H1_weight.data.uniform_(-0.1, 0.1)
        self.H1_bias.data.uniform_(-0.1, 0.1)

        self.L2_weight.data.uniform_(-0.1, 0.1)
        self.L2_bias.data.uniform_(-0.1, 0.1)
        
        
        self.V2_weight.data.uniform_(-0.1, 0.1)
        self.V2_bias.data.uniform_(-0.1, 0.1)
               
        self.H2_weight.data.uniform_(-0.1, 0.1)
        self.H2_bias.data.uniform_(-0.1, 0.1)
        
        self.pad = torch.nn.ConstantPad2d(kernel_size // 2,0)
         
    def forward(self, input):
#         x = self.matrix3d_matrix1d(input, self.kernel_size )
        # See the autograd section for explanation of what happens here.
#         res = test_fun.apply(input, self.L1_weight, self.L1_bias)
        
        out_L1 = conv1d_L.apply(input, self.L1_weight, self.L1_bias)
        # print('out_L1:', out_L1.size())
        
        out_V1 = conv1d_V.apply(out_L1, self.V1_weight, self.V1_bias)
        # print('out_V1:', out_V1.size())
        
        out_H1 = conv1d_H.apply(out_V1, self.H1_weight, self.H1_bias)
        # print('out_H1:', out_H1.size())
        
        out_L2 = conv1d_L.apply(out_H1, self.L2_weight, self.L2_bias)
        # print('out_L2:', out_L2.size())
        
        out_L2 = self.pad(out_L2)
        # print('out_L2 pading:', out_L2.size())
        
        out_V2 = conv1d_V.apply(out_L2, self.V2_weight, self.V2_bias)
        # print('out_V2:', out_V2.size())
        
        out_H2 = conv1d_H.apply(out_V2, self.H2_weight, self.H2_bias)
        # print('out_H2:', out_H2.size())
        return out_H2
#     def extra_repr(self):
#         # (Optional)Set the extra information about this module. You can test
#         # it by printing an object of this class.
#         return 'in_features={}, out_features={}, bias={}'.format(
#             self.in_features, self.out_features, self.bias is not None)


def get_shape(input_shape, weight_shape):
    batch_size, channels, row, col = list(input_shape)
    _, wight_size = list(weight_shape)
    
    half_ws = wight_size // 2
    windows_per = col - wight_size + 1
    
    new_col = windows_per
    
    XC_row = batch_size * row * windows_per
    XC_col = wight_size
    XC_shape = [XC_row, XC_col]

    output_shape = [batch_size, channels, row, new_col]
    return output_shape


def get_pad_shape(input_shape, weight_shape):
    out_channels, batch_size, row, col = list(input_shape)
    _, wight_size = list(weight_shape)
    pad_shape = [out_channels, batch_size, row , wight_size - 1]
    
    return pad_shape



def im2col(input, weight_shape):

    in_matrix =input
    out_channels, wight_size = list(weight_shape)
    
    XC_shape = [out_channels, -1, wight_size]

    len_row = in_matrix.size(3)
    half_ws = wight_size // 2
    index_start = half_ws
    index_end = len_row - half_ws

    XC = []
    for per_channel in in_matrix:
        per_channel = per_channel.reshape(-1,len_row)

        temp_res = []

        for i in per_channel:
            for index in range(index_start, index_end):
                temp  = i[index-half_ws : index+half_ws+1]
                temp_res.append(temp)

        temp_res = torch.cat(temp_res,dim = 0)
        temp_res = temp_res.reshape(-1, wight_size)
        XC.append(temp_res)

    XC = torch.cat(XC,dim = 0)
    XC = XC.reshape(*XC_shape)
    return XC

def XC_mul_WC(XC, weight, bias = None):
            
    out_matrix = []
    out_channels = weight.size(0)
    for i in range(out_channels):
        sub_weight = weight[i,:]
        sub_weight.unsqueeze_(1)
        sub_feature = XC[i,:,:]
        temp = sub_feature.mm(sub_weight)
        
        if bias is not None:
            sub_bias = bias[i]
            temp +=  sub_bias

        out_matrix.append(temp)

    out_matrix = torch.cat(out_matrix,dim=0)
    return out_matrix

class conv1d_L(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias):
        
        in_matrix =input.permute(0, 2, 3, 1)
        
        out_shape = list(in_matrix.size())
        out_shape = out_shape[:-1] + [-1]
        
        
        channels = in_matrix.size(3)

        XC = in_matrix.reshape(-1,channels)
        
        out_matrix = XC.mm(weight)
        out_matrix = out_matrix + bias
        out_matrix = out_matrix.reshape(*out_shape)
        out_matrix = out_matrix.permute(0, 3, 1, 2)
        
        ctx.save_for_backward(XC, weight, bias)

        return out_matrix



    @staticmethod
    def backward(ctx, grad_output):
#         print('conv1d_L:',ctx.needs_input_grad)

        XC, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        in_channels, out_channels = list(weight.size())
        
        dY = grad_output.permute(0, 2, 3, 1)
        dYC = dY.reshape(-1,out_channels)
        
        if ctx.needs_input_grad[0]:
        
            out_shape = list(dY.size())
            out_shape = out_shape[:-1] + [-1]

            grad_input = dYC.mm(weight.t())
            
            grad_input = grad_input.reshape(*out_shape)
            grad_input =grad_input.permute(0, 3, 1, 2)
            
        if ctx.needs_input_grad[1]:
            grad_weight = XC.t().mm(dYC)
                                
        if ctx.needs_input_grad[2]:
            grad_bias = dYC.sum(dim = 0)
            grad_bias = grad_bias.view_as(bias)
            
        return grad_input, grad_weight, grad_bias

class conv1d_V(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias):
        
        out_channels, _ = list(weight.size())
        
        
        in_matrix =input.permute(1, 0, 3, 2)
        out_shape = get_shape(in_matrix.size(), weight.size())
        
        
        XC = im2col(in_matrix, weight.size())
        out_matrix = XC_mul_WC(XC, weight, bias)
        
        out_matrix = out_matrix.reshape(*out_shape)
        out_matrix =out_matrix.permute(1, 0, 3, 2)
        
        ctx.save_for_backward(XC, weight, bias)
        
        return out_matrix



    @staticmethod
    def backward(ctx, grad_output):
        
#         print('conv1d_H:',ctx.needs_input_grad)
        ### init
        XC, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        out_channels, wight_size = list(weight.size())
        
        dY = grad_output.permute(1, 0, 3, 2)
    
 
        if ctx.needs_input_grad[0]:
            
            grad_input = []
            

            weight_flip = torch.flip(weight, dims = [1])
            
            pad_shape = get_pad_shape(dY.size(), weight.size())
            pad = torch.zeros(*pad_shape)
            pad = pad.to(device)


            dY_pad = torch.cat([pad, dY, pad],dim = 3)
            dYC_pad =  im2col(dY_pad, weight.size())
            
            out_shape = get_shape(dY_pad.size(), weight.size())
            
            grad_input = XC_mul_WC(dYC_pad, weight_flip)
            grad_input = grad_input.reshape(*out_shape)
            grad_input =grad_input.permute(1, 0, 3, 2)
            

            
        if ctx.needs_input_grad[1]:
            grad_weight = []

            for i in range(out_channels):
                sub_XC = XC[i,:,:]
                sub_dY = dY[i,:,:,:]
                sub_dY = sub_dY.reshape(-1,1)
                grad_weight.append(sub_XC.t().mm(sub_dY))

            grad_weight = torch.cat(grad_weight)
            grad_weight = grad_weight.reshape(out_channels, wight_size)
            

            
        if ctx.needs_input_grad[2]:
            
            grad_bias = dY.reshape(out_channels,-1)
            grad_bias = grad_bias.sum(dim = 1)
            grad_bias = grad_bias.view_as(bias)
            
#         print('V grad_input',grad_input)
#         print('V grad_weight',grad_weight)
#         print('V grad_bias',grad_bias)

        return grad_input, grad_weight, grad_bias
    




class conv1d_H(torch.autograd.Function):
    

    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias):

        out_channels, _ = list(weight.size())
        
        
        in_matrix =input.permute(1, 0, 2, 3)
        out_shape = get_shape(in_matrix.size(), weight.size())
        
        
        XC = im2col(in_matrix, weight.size())
        out_matrix = XC_mul_WC(XC, weight, bias)
        
        out_matrix = out_matrix.reshape(*out_shape)
        out_matrix =out_matrix.permute(1, 0, 2, 3)
        
        ctx.save_for_backward(XC, weight, bias)
        
        return out_matrix



    @staticmethod
    def backward(ctx, grad_output):
        
#         print('conv1d_H:',ctx.needs_input_grad)
        XC, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        out_channels, wight_size = list(weight.size())
        
        dY = grad_output.permute(1, 0, 2, 3)
        # out_channels, batch_size, row, col

        
        if ctx.needs_input_grad[0]:
            
            grad_input = []
            

            weight_flip = torch.flip(weight, dims = [1])

            pad_shape = get_pad_shape(dY.size(), weight.size())
            pad = torch.zeros(*pad_shape)
            pad = pad.to(device)


            dY_pad = torch.cat([pad, dY, pad],dim = 3)
            dYC_pad =  im2col(dY_pad, weight.size())
            
            
            out_shape = get_shape(dY_pad.size(), weight.size())
            
            grad_input = XC_mul_WC(dYC_pad, weight_flip)
            grad_input = grad_input.reshape(*out_shape)
            grad_input =grad_input.permute(1, 0, 2, 3)   
            

            
        if ctx.needs_input_grad[1]:
            grad_weight = []

            for i in range(out_channels):
                sub_XC = XC[i,:,:]
                sub_dY = dY[i,:,:,:]
                sub_dY = sub_dY.reshape(-1,1)
                grad_weight.append(sub_XC.t().mm(sub_dY))

            grad_weight = torch.cat(grad_weight)
            grad_weight = grad_weight.reshape(out_channels, wight_size)
            

            
        if ctx.needs_input_grad[2]:
            
            grad_bias = dY.reshape(out_channels,-1)
            grad_bias = grad_bias.sum(dim = 1)
            grad_bias = grad_bias.view_as(bias)
            
            
#         print('H grad_input',grad_input)
#         print('H grad_weight',grad_weight)
#         print('H grad_bias',grad_bias)

        return grad_input, grad_weight, grad_bias
    
