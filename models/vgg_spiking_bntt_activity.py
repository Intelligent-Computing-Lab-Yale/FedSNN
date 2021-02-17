# File : model_cifar10.py
# Descr: Define SNN models for the CIFAR10 dataset
# Date : March 22, 2019

# --------------------------------------------------
# Imports
# --------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import numpy as np
import numpy.linalg as LA
from torch.autograd import Variable


# --------------------------------------------------
# Spiking neuron with fast-sigmoid surrogate gradient
# This class is replicated from:
# https://github.com/fzenke/spytorch/blob/master/notebooks/SpyTorchTutorial2.ipynb
# --------------------------------------------------
class SuperSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid as
    was done in Zenke & Ganguli (2018).
    """
    scale = 100.0  # Controls the steepness of the fast-sigmoid surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass, we compute a step function of the input Tensor and
        return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use
        the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass, we receive a Tensor we need to compute
        the surrogate gradient of the loss with respect to the input.
        Here we use the normalized negative part of a fast sigmoid
        as was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (SuperSpike.scale * torch.abs(input) + 1.0) ** 2
        return grad


# --------------------------------------------------
# Spiking neuron with piecewise-linear surrogate gradient
# --------------------------------------------------
class LinearSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    """
    gamma = 0.3  # Controls the dampening of the piecewise-linear surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass, we compute a step function of the input Tensor and
        return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use
        the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass, we receive a Tensor we need to compute
        the surrogate gradient of the loss with respect to the input.
        Here we use the piecewise-linear surrogate gradient as was
        done in Bellec et al. (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * LinearSpike.gamma * F.threshold(1.0 - torch.abs(input), 0, 0)
        return grad


# --------------------------------------------------
# Spiking neuron with exponential surrogate gradient
# --------------------------------------------------
class ExpSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the exponential surrogate gradient as was done in
    Shrestha et al. (2018).
    """
    alpha = 1.0  # Controls the magnitude of the exponential surrogate gradient
    beta = 10.0  # Controls the steepness of the exponential surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass, we compute a step function of the input Tensor and
        return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use
        the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass, we receive a Tensor we need to compute
        the surrogate gradient of the loss with respect to the input.
        Here we use the exponential surrogate gradient as was done
        in Shrestha et al. (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * ExpSpike.alpha * torch.exp(-ExpSpike.beta * torch.abs(input))
        return grad


# --------------------------------------------------
# Spiking neuron with pass-through surrogate gradient
# --------------------------------------------------
class PassThruSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the pass-through surrogate gradient.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass, we compute a step function of the input Tensor and
        return it. For this spiking nonlinearity, the context object ctx does not
        stash input information since it is not used for backpropagation.
        """
        # ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass, we receive a Tensor we need to compute
        the surrogate gradient of the loss with respect to the input.
        Here we use the pass-through surrogate gradient.
        """
        # input,   = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input


# Overwrite the naive spike function by differentiable spiking nonlinearity which implements a surrogate gradient
def init_spike_fn(grad_type):
    if (grad_type == 'FastSigm'):
        spike_fn = SuperSpike.apply
    elif (grad_type == 'Linear'):
        spike_fn = LinearSpike.apply
    elif (grad_type == 'Exp'):
        spike_fn = ExpSpike.apply
    elif (grad_type == 'PassThru'):
        spike_fn = PassThruSpike.apply
    else:
        sys.exit("Unknown gradient type '{}'".format(grad_type))
    return spike_fn


# --------------------------------------------------
# Poisson spike generator
#   Positive spike is generated (i.e.  1 is returned) if rand()<=abs(input) and sign(input)= 1
#   Negative spike is generated (i.e. -1 is returned) if rand()<=abs(input) and sign(input)=-1
# --------------------------------------------------
def PoissonGen(inp, rescale_fac=2.0):
    rand_inp = torch.rand_like(inp).cuda()
    return torch.mul(torch.le(rand_inp * rescale_fac, torch.abs(inp)).float(), torch.sign(inp))








class SNN_VGG9_TBN(nn.Module):
    def __init__(self, dt=0.001, t_end=0.100, inp_rate=100, grad_type='Linear', thresh_init_wnorm=False,
                 leak_mem=0.99, img_size=32, inp_maps=3, c1_maps=64, c2_maps=64, ksize=3, fc0_size=200,
                 num_cls=1000, drop_rate=0.5, use_max_out_over_time=False, timesteps = 20):
        super(SNN_VGG9_TBN, self).__init__()

        # ConvSNN architecture parameters
        self.img_size = img_size
        self.inp_maps = inp_maps
        self.c1_maps = 64
        self.c1_dim = self.img_size
        self.c2_maps = 128
        self.c3_maps = 256
        self.c4_maps = 512


        self.ksize = ksize
        self.fc0_size = fc0_size
        self.num_cls = num_cls

        # ConvSNN simulation parameters
        self.dt = dt
        self.t_end = t_end
        # self.num_steps = int(self.t_end / self.dt)
        self.num_steps = timesteps
        self.inp_rate = inp_rate
        self.inp_rescale_fac = 1.0 / (self.dt * self.inp_rate)
        self.grad_type = grad_type
        self.grad_type_pool = 'PassThru'
        self.thresh_init_wnorm = thresh_init_wnorm
        self.leak_mem = leak_mem#0.95  # leak_mem
        self.drop_rate = drop_rate
        self.lnorm_ord = 2
        self.scale_thresh = 1.0
        self.use_max_out_over_time = use_max_out_over_time

        self.dropout_layer = nn.Dropout2d(p=0.2)

        self.one_stamp = 1
        self.batch_num = self.num_steps // self.one_stamp

        print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print ("***** time step per batchnorm", self.batch_num)
        print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        affine_flag = True
        bias_flag = False
        # Instantiate the ConvSNN layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bn1_list = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv1_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bn1_1_list = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool1 = nn.AvgPool2d(kernel_size=2)  # Default stride = kernel_size

        self.conv2 = nn.Conv2d(64, 128, kernel_size=self.ksize, stride=1, padding=1, bias=bias_flag)
        self.bn2_list = nn.ModuleList([nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv3 = nn.Conv2d(128, 128, kernel_size=self.ksize, stride=1, padding=1, bias=bias_flag)
        self.bn3_list = nn.ModuleList([nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool2 = nn.AvgPool2d(kernel_size=2)  # Default stride = kernel_size

        self.conv4 = nn.Conv2d(128, 256, kernel_size=self.ksize, stride=1, padding=1, bias=bias_flag)
        self.bn4_list = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv5 = nn.Conv2d(256, 256, kernel_size=self.ksize, stride=1, padding=1, bias=bias_flag)
        self.bn5_list = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv6 = nn.Conv2d(256, 256, kernel_size=self.ksize, stride=1, padding=1, bias=bias_flag)
        self.bn6_list = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool3 = nn.AvgPool2d(kernel_size=2)  # Default stride = kernel_size



        self.drop = nn.Dropout(p=0.2)

        self.fc1 = nn.Linear((self.img_size//8)*(self.img_size //8)*256, 2*2*256, bias=bias_flag)
        self.bnfc_list = nn.ModuleList([nn.BatchNorm1d( 2*2*256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])

        self.fc2 = nn.Linear(2*2*256, self.num_cls, bias=bias_flag)

        batchnormlist = [self.bn1_list, self.bn1_1_list, self.bn2_list, self.bn3_list, self.bn4_list, self.bn5_list,
                         self.bn6_list, self.bnfc_list]

        #TODO turn off bias of batchnorm
        for bnlist in batchnormlist:
            for bnbn in bnlist:
                bnbn.bias = None


        # Initialize the firing thresholds of all the layers
        for m in self.modules():
            if (isinstance(m, nn.Conv2d)):
                m.threshold = 1.0
                # torch.nn.init.kaiming_normal_(m.weight,a=1)
                torch.nn.init.xavier_uniform_(m.weight, gain=2)

            elif (isinstance(m, nn.AvgPool2d)):
                m.threshold = 0.75
            elif (isinstance(m, nn.Linear)):
                m.threshold = 1.0
                # torch.nn.init.kaiming_normal_(m.weight,a=1)
                torch.nn.init.xavier_uniform_(m.weight, gain=2)


        # Instantiate differentiable spiking nonlinearity
        self.spike_fn = init_spike_fn(self.grad_type)
        self.spike_pool = init_spike_fn(self.grad_type_pool)

    def fc_init(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)

        torch.nn.init.xavier_uniform_(self.fc2.weight)


    def forward(self, inp, count_active_layers = False, report_activity = False):

        active_layer_count = 9
        if count_active_layers == True:
            return active_layer_count
        activity = torch.zeros(active_layer_count).cuda()

        # avg_spike_time = []
        # Initialize the neuronal membrane potentials and dropout masks
        batch_size = inp.size(0)
        mem_conv1 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv1_1 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv2 = torch.zeros(batch_size, 128, self.img_size//2, self.img_size//2).cuda()
        mem_conv3 = torch.zeros(batch_size, 128, self.img_size//2, self.img_size//2).cuda()
        mem_conv4 = torch.zeros(batch_size, 256, self.img_size//4, self.img_size//4).cuda()
        mem_conv5 = torch.zeros(batch_size, 256, self.img_size//4, self.img_size//4).cuda()
        mem_conv6 = torch.zeros(batch_size, 256, self.img_size//4, self.img_size//4).cuda()



        mem_fc1 = torch.zeros(batch_size, 1024).cuda()
        mem_fc2 = torch.zeros(batch_size, self.num_cls).cuda()

        fc_dropout_mask = self.drop(torch.ones([batch_size, 1024]).cuda())

        for t in range(self.num_steps):
            spike_inp = PoissonGen(inp)
            activity[0] += torch.count_nonzero(spike_inp.detach())/torch.numel(spike_inp.detach())
            # # spike_x = torch.cat([spike_inp]*22,1)[:,:64,:,:]
            # out_prev = spike_inp
            #
            # Compute the conv1 outputs
            mem_thr   = (mem_conv1/self.conv1.threshold) - 1.0
            out       = self.spike_fn(mem_thr)
            rst       = torch.zeros_like(mem_conv1).cuda()
            rst[mem_thr>0] = self.conv1.threshold
            mem_conv1 = (self.leak_mem*mem_conv1 + self.bn1_list[int(t/self.one_stamp)](self.conv1(spike_inp)) -rst)
            out_prev  = out.clone()

            activity[1] += torch.count_nonzero(out_prev.detach())/torch.numel(out_prev.detach())
            # Compute the conv1_1 outputs
            mem_thr = (mem_conv1_1 / self.conv1_1.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv1_1).cuda()
            rst[mem_thr > 0] = self.conv1_1.threshold
            mem_conv1_1 = (self.leak_mem * mem_conv1_1 + self.bn1_1_list[int(t/self.one_stamp)](self.conv1_1(out_prev)) - rst)
            out_prev = out.clone()


            # Compute the avgpool1 outputs
            out =  self.pool1(out_prev)
            out_prev = out.clone()

            # mem_thr = (mem_pool1 / self.pool1.threshold) - 1.0
            # out = self.spike_pool(mem_thr)
            # rst = torch.zeros_like(mem_pool1).cuda()
            # rst[mem_thr > 0] = self.pool1.threshold
            # mem_pool1 = mem_pool1 + self.pool1(out_prev) - rst
            # out_prev = out.clone()


            activity[2] += torch.count_nonzero(out_prev.detach())/torch.numel(out_prev.detach())
            # Compute the conv2 outputs
            mem_thr   = (mem_conv2/self.conv2.threshold) - 1.0
            out       = self.spike_fn(mem_thr)
            rst       = torch.zeros_like(mem_conv2).cuda()
            rst[mem_thr>0] = self.conv2.threshold
            mem_conv2 = (self.leak_mem*mem_conv2 + self.bn2_list[int(t/self.one_stamp)](self.conv2(out_prev)) -rst)
            out_prev  = out.clone()


            activity[3] += torch.count_nonzero(out_prev.detach())/torch.numel(out_prev.detach())
            # Compute the conv3 outputs
            mem_thr = (mem_conv3 / self.conv3.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv3).cuda()
            rst[mem_thr > 0] = self.conv3.threshold
            mem_conv3 = (self.leak_mem * mem_conv3 + self.bn3_list[int(t/self.one_stamp)](self.conv3(out_prev)) - rst)
            out_prev = out.clone()

            # Compute the avgpool2 outputs
            out = self.pool2(out_prev)
            out_prev = out.clone()
            # mem_thr = (mem_pool2 / self.pool2.threshold) - 1.0
            # out = self.spike_pool(mem_thr)
            # rst = torch.zeros_like(mem_pool2).cuda()
            # rst[mem_thr > 0] = self.pool2.threshold
            # mem_pool2 = mem_pool2 + self.pool2(out_prev) - rst
            # out_prev = out.clone()


            activity[4] += torch.count_nonzero(out_prev.detach())/torch.numel(out_prev.detach())
            # Compute the conv4 outputs
            mem_thr = (mem_conv4 / self.conv4.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv4).cuda()
            rst[mem_thr > 0] = self.conv4.threshold
            mem_conv4 = (self.leak_mem * mem_conv4 + self.bn4_list[int(t/self.one_stamp)](self.conv4(out_prev)) - rst)
            out_prev = out.clone()


            activity[5] += torch.count_nonzero(out_prev.detach())/torch.numel(out_prev.detach())
            # Compute the conv5 outputs
            mem_thr = (mem_conv5 / self.conv5.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv5).cuda()
            rst[mem_thr > 0] = self.conv5.threshold
            mem_conv5 = (self.leak_mem * mem_conv5 + self.bn5_list[int(t/self.one_stamp)](self.conv5(out_prev)) - rst)
            out_prev = out.clone()


            activity[6] += torch.count_nonzero(out_prev.detach())/torch.numel(out_prev.detach())
            # Compute the conv6 outputs
            mem_thr = (mem_conv6 / self.conv6.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv6).cuda()
            rst[mem_thr > 0] = self.conv6.threshold
            mem_conv6 = (self.leak_mem * mem_conv6 + self.bn6_list[int(t/self.one_stamp)](self.conv6(out_prev)) - rst)
            out_prev = out.clone()

            # Compute the avgpool3 outputs
            out = self.pool3(out_prev)
            out_prev = out.clone()
            # mem_thr = (mem_pool3 / self.pool3.threshold) - 1.0
            # out = self.spike_pool(mem_thr)
            # rst = torch.zeros_like(mem_pool3).cuda()
            # rst[mem_thr > 0] = self.pool3.threshold
            # mem_pool3 = mem_pool3 + self.pool3(out_prev) - rst
            # out_prev = out.clone()


            activity[7] += torch.count_nonzero(out_prev.detach())/torch.numel(out_prev.detach())
            out_prev = out_prev.reshape(batch_size, -1)
            # compute fc1
            mem_thr = (mem_fc1 / self.fc1.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_fc1).cuda()
            rst[mem_thr > 0] = self.fc1.threshold
            mem_fc1 = (self.leak_mem * mem_fc1 + self.bnfc_list[int(t/self.one_stamp)](self.fc1(out_prev)) - rst)
            # mem_fc1 = (self.leak_mem * mem_fc1 + (self.fc1(out_prev)) - rst)

            out_prev = out.clone()

            # out_prev = fc_dropout_mask *out_prev

            # # TODO last spike expectation
            # avg_spike = out_prev.sum(1).sum(0) / out_prev.size(1) / out_prev.size(0)
            # avg_spike_time.append(float(avg_spike.cpu().data.numpy()))

            activity[8] += torch.count_nonzero(out_prev.detach())/torch.numel(out_prev.detach())
            # compute fc1
            mem_fc2 = (1 * mem_fc2 + self.fc2(out_prev))

        print(activity)
        activity = [x / self.num_steps for x in activity]
        print(activity)
        if report_activity:
            return activity

        out_voltage  = mem_fc2
        out_voltage = (out_voltage) / self.num_steps


        return out_voltage
# ----------------------------------------------







class SNN_VGG16_TBN(nn.Module):
    def __init__(self, dt=0.001, t_end=0.100, inp_rate=100, grad_type='Linear', thresh_init_wnorm=False,
                 leak_mem=0.99, img_size=32, inp_maps=3, c1_maps=64, c2_maps=64, ksize=3, fc0_size=200,
                 num_cls=1000, drop_rate=0.5, use_max_out_over_time=False, timesteps = 20):
        super(SNN_VGG16_TBN, self).__init__()

        # ConvSNN architecture parameters
        self.img_size = img_size
        self.inp_maps = inp_maps
        self.c1_maps = 64
        self.c1_dim = self.img_size
        self.c2_maps = 128
        self.c3_maps = 256
        self.c4_maps = 512


        self.ksize = ksize
        self.fc0_size = fc0_size
        self.num_cls = num_cls

        # ConvSNN simulation parameters
        self.dt = dt
        self.t_end = t_end
        # self.num_steps = int(self.t_end / self.dt)
        self.num_steps = timesteps
        self.inp_rate = inp_rate
        self.inp_rescale_fac = 1.0 / (self.dt * self.inp_rate)
        self.grad_type = grad_type
        self.grad_type_pool = 'PassThru'
        self.thresh_init_wnorm = thresh_init_wnorm
        self.leak_mem = 0.99  # leak_mem
        self.drop_rate = drop_rate
        self.lnorm_ord = 2
        self.scale_thresh = 1.0
        self.use_max_out_over_time = use_max_out_over_time

        self.dropout_layer = nn.Dropout2d(p=0.2)

        self.one_stamp = 1
        self.batch_num = self.num_steps // self.one_stamp

        print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print ("***** time step per batchnorm", self.batch_num)
        print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        affine_flag = True


        # Instantiate the ConvSNN layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_list = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv1_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_1_list = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool1 = nn.AvgPool2d(kernel_size=2)  # Default stride = kernel_size

        self.conv2 = nn.Conv2d(64, 128, kernel_size=self.ksize, stride=1, padding=1, bias=False)
        self.bn2_list = nn.ModuleList([nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv3 = nn.Conv2d(128, 128, kernel_size=self.ksize, stride=1, padding=1, bias=False)
        self.bn3_list = nn.ModuleList([nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool2 = nn.AvgPool2d(kernel_size=2)  # Default stride = kernel_size

        self.conv4 = nn.Conv2d(128, 256, kernel_size=self.ksize, stride=1, padding=1, bias=False)
        self.bn4_list = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv5 = nn.Conv2d(256, 256, kernel_size=self.ksize, stride=1, padding=1, bias=False)
        self.bn5_list = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv6 = nn.Conv2d(256, 256, kernel_size=self.ksize, stride=1, padding=1, bias=False)
        self.bn6_list = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool3 = nn.AvgPool2d(kernel_size=2)

        self.conv7 = nn.Conv2d(256, 512, kernel_size=self.ksize, stride=1, padding=1, bias=False)
        self.bn7_list = nn.ModuleList([nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv8 = nn.Conv2d(512, 512, kernel_size=self.ksize, stride=1, padding=1, bias=False)
        self.bn8_list = nn.ModuleList([nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv9 = nn.Conv2d(512, 512, kernel_size=self.ksize, stride=1, padding=1, bias=False)
        self.bn9_list = nn.ModuleList([nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool4 = nn.AvgPool2d(kernel_size=2)

        self.conv10 = nn.Conv2d(512, 512, kernel_size=self.ksize, stride=1, padding=1, bias=False)
        self.bn10_list = nn.ModuleList([nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv11 = nn.Conv2d(512, 512, kernel_size=self.ksize, stride=1, padding=1, bias=False)
        self.bn11_list = nn.ModuleList([nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv12 = nn.Conv2d(512, 512, kernel_size=self.ksize, stride=1, padding=1, bias=False)
        self.bn12_list = nn.ModuleList([nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool5 = nn.AdaptiveAvgPool2d((1,1))



        self.fc1 = nn.Linear(512, 4096, bias=False)
        self.bnfc_list = nn.ModuleList([nn.BatchNorm1d( 4096, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])

        self.fc2 = nn.Linear(4096, self.num_cls, bias=False)


        batchnormlist = [self.bn1_list, self.bn1_1_list, self.bn2_list, self.bn3_list, self.bn4_list, self.bn5_list,
                         self.bn6_list, self.bn7_list,self.bn8_list,self.bn9_list,self.bn10_list, self.bn11_list,self.bn12_list, self.bnfc_list]

        # TODO turn off bias of batchnorm
        for bnlist in batchnormlist:
            for bnbn in bnlist:
                bnbn.bias = None

        # Initialize the firing thresholds of all the layers
        for m in self.modules():
            if (isinstance(m, nn.Conv2d)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)
            elif (isinstance(m, nn.AvgPool2d)):
                m.threshold = 0.75
            elif (isinstance(m, nn.Linear)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)
                if (self.thresh_init_wnorm):
                    lnorm = LA.norm(m.weight.data, self.lnorm_ord)
                    thresh_init = lnorm * self.scale_thresh
                    m.threshold = torch.from_numpy(np.array([thresh_init])).float().cuda()
                    print('Wl{}norm: {:.2f}; Threshold: {:.2f}\n'.format(self.lnorm_ord, lnorm, m.threshold[0]))

        # Instantiate differentiable spiking nonlinearity
        self.spike_fn = init_spike_fn(self.grad_type)
        self.spike_pool = init_spike_fn(self.grad_type_pool)

    def fc_init(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)

        torch.nn.init.xavier_uniform_(self.fc2.weight)


    def forward(self, inp):
        # avg_spike_time = []
        # Initialize the neuronal membrane potentials and dropout masks
        batch_size = inp.size(0)
        mem_conv1 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv1_1 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()

        mem_conv2 = torch.zeros(batch_size, 128, self.img_size//2, self.img_size//2).cuda()
        mem_conv3 = torch.zeros(batch_size, 128, self.img_size//2, self.img_size//2).cuda()

        mem_conv4 = torch.zeros(batch_size, 256, self.img_size//4, self.img_size//4).cuda()
        mem_conv5 = torch.zeros(batch_size, 256, self.img_size//4, self.img_size//4).cuda()
        mem_conv6 = torch.zeros(batch_size, 256, self.img_size//4, self.img_size//4).cuda()

        mem_conv7 = torch.zeros(batch_size, 512, self.img_size//8, self.img_size//8).cuda()
        mem_conv8 = torch.zeros(batch_size, 512, self.img_size//8, self.img_size//8).cuda()
        mem_conv9 = torch.zeros(batch_size, 512, self.img_size//8, self.img_size//8).cuda()

        mem_conv10 = torch.zeros(batch_size, 512, self.img_size//16, self.img_size//16).cuda()
        mem_conv11 = torch.zeros(batch_size, 512, self.img_size//16, self.img_size//16).cuda()
        mem_conv12 = torch.zeros(batch_size, 512, self.img_size//16, self.img_size//16).cuda()





        mem_fc1 = torch.zeros(batch_size, 4096).cuda()
        mem_fc2 = torch.zeros(batch_size, self.num_cls).cuda()


        for t in range(self.num_steps):
            spike_inp = PoissonGen(inp)
            # spike_x = torch.cat([spike_inp]*22,1)[:,:64,:,:]
            out_prev = spike_inp

            # Compute the conv1 outputs
            mem_thr   = (mem_conv1/self.conv1.threshold) - 1.0
            out       = self.spike_fn(mem_thr)
            rst       = torch.zeros_like(mem_conv1).cuda()
            rst[mem_thr>0] = self.conv1.threshold
            mem_conv1 = (self.leak_mem*mem_conv1 + self.bn1_list[int(t/self.one_stamp)](self.conv1(out_prev)) -rst)
            out_prev  = out.clone()

            # Compute the conv1_1 outputs
            mem_thr = (mem_conv1_1 / self.conv1_1.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv1_1).cuda()
            rst[mem_thr > 0] = self.conv1_1.threshold
            mem_conv1_1 = (self.leak_mem * mem_conv1_1 + self.bn1_1_list[int(t/self.one_stamp)](self.conv1_1(out_prev)) - rst)
            out_prev = out.clone()


            # Compute the avgpool1 outputs
            out = self.pool1(out_prev)
            out_prev = out.clone()



            # Compute the conv2 outputs
            mem_thr   = (mem_conv2/self.conv2.threshold) - 1.0
            out       = self.spike_fn(mem_thr)
            rst       = torch.zeros_like(mem_conv2).cuda()
            rst[mem_thr>0] = self.conv2.threshold
            mem_conv2 = (self.leak_mem*mem_conv2 + self.bn2_list[int(t/self.one_stamp)](self.conv2(out_prev)) -rst)
            out_prev  = out.clone()

            # Compute the conv3 outputs
            mem_thr = (mem_conv3 / self.conv3.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv3).cuda()
            rst[mem_thr > 0] = self.conv3.threshold
            mem_conv3 = (self.leak_mem * mem_conv3 + self.bn3_list[int(t/self.one_stamp)](self.conv3(out_prev)) - rst)
            out_prev = out.clone()

            # Compute the avgpool2 outputs
            out = self.pool2(out_prev)
            out_prev = out.clone()

            # Compute the conv4 outputs
            mem_thr = (mem_conv4 / self.conv4.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv4).cuda()
            rst[mem_thr > 0] = self.conv4.threshold
            mem_conv4 = (self.leak_mem * mem_conv4 + self.bn4_list[int(t/self.one_stamp)](self.conv4(out_prev)) - rst)
            out_prev = out.clone()

            # Compute the conv5 outputs
            mem_thr = (mem_conv5 / self.conv5.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv5).cuda()
            rst[mem_thr > 0] = self.conv5.threshold
            mem_conv5 = (self.leak_mem * mem_conv5 + self.bn5_list[int(t/self.one_stamp)](self.conv5(out_prev)) - rst)
            out_prev = out.clone()

            # Compute the conv6 outputs
            mem_thr = (mem_conv6 / self.conv6.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv6).cuda()
            rst[mem_thr > 0] = self.conv6.threshold
            mem_conv6 = (self.leak_mem * mem_conv6 + self.bn6_list[int(t/self.one_stamp)](self.conv6(out_prev)) - rst)
            out_prev = out.clone()

            # Compute the avgpool3 outputs
            out = self.pool3(out_prev)
            out_prev = out.clone()



            # Compute the conv7 outputs
            mem_thr = (mem_conv7 / self.conv7.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv7).cuda()
            rst[mem_thr > 0] = self.conv7.threshold
            mem_conv7 = (self.leak_mem * mem_conv7 + self.bn7_list[int(t / self.one_stamp)](self.conv7(out_prev)) - rst)
            out_prev = out.clone()

            # Compute the conv8 outputs
            mem_thr = (mem_conv8 / self.conv8.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv8).cuda()
            rst[mem_thr > 0] = self.conv8.threshold
            mem_conv8 = (self.leak_mem * mem_conv8 + self.bn8_list[int(t / self.one_stamp)](self.conv8(out_prev)) - rst)
            out_prev = out.clone()

            # Compute the conv9 outputs
            mem_thr = (mem_conv9 / self.conv9.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv9).cuda()
            rst[mem_thr > 0] = self.conv9.threshold
            mem_conv9 = (self.leak_mem * mem_conv9 + self.bn9_list[int(t / self.one_stamp)](self.conv9(out_prev)) - rst)
            out_prev = out.clone()

            # Compute the avgpool4 outputs
            out = self.pool4(out_prev)
            out_prev = out.clone()



            # Compute the conv10 outputs
            mem_thr = (mem_conv10 / self.conv10.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv10).cuda()
            rst[mem_thr > 0] = self.conv10.threshold
            mem_conv10 = (self.leak_mem * mem_conv10 + self.bn10_list[int(t / self.one_stamp)](self.conv10(out_prev)) - rst)
            out_prev = out.clone()

            # Compute the conv11 outputs
            mem_thr = (mem_conv11 / self.conv11.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv11).cuda()
            rst[mem_thr > 0] = self.conv11.threshold
            mem_conv11 = (self.leak_mem * mem_conv11 + self.bn11_list[int(t / self.one_stamp)](self.conv11(out_prev)) - rst)
            out_prev = out.clone()

            # Compute the conv12 outputs
            mem_thr = (mem_conv12 / self.conv12.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv12).cuda()
            rst[mem_thr > 0] = self.conv12.threshold
            mem_conv12 = (self.leak_mem * mem_conv12 + self.bn12_list[int(t / self.one_stamp)](self.conv12(out_prev)) - rst)
            out_prev = out.clone()

            # Compute the avgpool5 outputs
            out = self.pool5(out_prev)
            out_prev = out.clone()

            out_prev = out_prev.reshape(batch_size, -1)

            # compute fc1
            mem_thr = (mem_fc1 / self.fc1.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_fc1).cuda()
            rst[mem_thr > 0] = self.fc1.threshold
            mem_fc1 = (self.leak_mem * mem_fc1 + self.bnfc_list[int(t/self.one_stamp)](self.fc1(out_prev)) - rst)
            out_prev = out.clone()

            # # TODO last spike expectation
            # avg_spike = out_prev.sum(1).sum(0) / out_prev.size(1) / out_prev.size(0)
            # avg_spike_time.append(float(avg_spike.cpu().data.numpy()))

            # compute fc1
            mem_fc2 = (1 * mem_fc2 + self.fc2(out_prev))



        out_voltage  = mem_fc2
        out_voltage = (out_voltage) / self.num_steps


        return out_voltage
# ----------------------------------------------


# --------------------------------------------------
# Define a class for recording the SNN train/test loss.
# This class is replicated from Chankyu Lee's SNN-backprop code.
# --------------------------------------------------
class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count






class SNN_VGG11_TBN(nn.Module):
    def __init__(self, dt=0.001, t_end=0.100, inp_rate=100, grad_type='Linear', thresh_init_wnorm=False,
                 leak_mem=0.99, img_size=32, inp_maps=3, c1_maps=64, c2_maps=64, ksize=3, fc0_size=200,
                 num_cls=1000, drop_rate=0.5, use_max_out_over_time=False):
        super(SNN_VGG11_TBN, self).__init__()

        # ConvSNN architecture parameters
        self.img_size = img_size
        self.inp_maps = inp_maps
        self.c1_maps = 64
        self.c1_dim = self.img_size
        self.c2_maps = 128
        self.c3_maps = 256
        self.c4_maps = 512


        self.ksize = ksize
        self.fc0_size = fc0_size
        self.num_cls = num_cls

        # ConvSNN simulation parameters
        self.dt = dt
        self.t_end = t_end
        self.num_steps = int(self.t_end / self.dt)
        self.inp_rate = inp_rate
        self.inp_rescale_fac = 1.0 / (self.dt * self.inp_rate)
        self.grad_type = grad_type
        self.grad_type_pool = 'PassThru'
        self.thresh_init_wnorm = thresh_init_wnorm
        self.leak_mem = leak_mem#0.95  # leak_mem
        self.drop_rate = drop_rate
        self.lnorm_ord = 2
        self.scale_thresh = 1.0
        self.use_max_out_over_time = use_max_out_over_time

        self.dropout_layer = nn.Dropout2d(p=0.2)

        self.one_stamp = 1
        self.batch_num = self.num_steps // self.one_stamp

        print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print ("***** time step per batchnorm", self.batch_num)
        print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        affine_flag = True
        bias_flag = False
        # Instantiate the ConvSNN layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bn1_list = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool1 = nn.AvgPool2d(kernel_size=2)  # Default stride = kernel_size

        self.conv2 = nn.Conv2d(64, 128, kernel_size=self.ksize, stride=1, padding=1, bias=bias_flag)
        self.bn2_list = nn.ModuleList([nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool2 = nn.AvgPool2d(kernel_size=2)  # Default stride = kernel_size


        self.conv3 = nn.Conv2d(128, 256, kernel_size=self.ksize, stride=1, padding=1, bias=bias_flag)
        self.bn3_list = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv4 = nn.Conv2d(256, 256, kernel_size=self.ksize, stride=1, padding=1, bias=bias_flag)
        self.bn4_list = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool3 = nn.AvgPool2d(kernel_size=2)  # Default stride = kernel_size

        self.conv5 = nn.Conv2d(256, 512, kernel_size=self.ksize, stride=1, padding=1, bias=bias_flag)
        self.bn5_list = nn.ModuleList([nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv6 = nn.Conv2d(512, 512, kernel_size=self.ksize, stride=1, padding=1, bias=bias_flag)
        self.bn6_list = nn.ModuleList([nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool4 = nn.AvgPool2d(kernel_size=2)  # Default stride = kernel_size

        self.conv7 = nn.Conv2d(512, 512, kernel_size=self.ksize, stride=1, padding=1, bias=bias_flag)
        self.bn7_list = nn.ModuleList([nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv8 = nn.Conv2d(512, 512, kernel_size=self.ksize, stride=1, padding=1, bias=bias_flag)
        self.bn8_list = nn.ModuleList([nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool5 = nn.AvgPool2d(kernel_size=2)  # Default stride = kernel_size

        self.drop = nn.Dropout(p=0.2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        # self.fc1 = nn.Linear((self.img_size//32)*(self.img_size //32)*512, 4*2*2*256, bias=bias_flag)
        self.fc1 = nn.Linear(512, 4*2*2*256, bias=bias_flag)

        self.bnfc_list = nn.ModuleList([nn.BatchNorm1d( 4*2*2*256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])

        self.fc2 = nn.Linear(4*2*2*256, self.num_cls, bias=bias_flag)

        batchnormlist = [self.bn1_list, self.bn2_list, self.bn3_list, self.bn4_list, self.bn5_list,
                         self.bn6_list, self.bn7_list,self.bn8_list,self.bnfc_list]

        #TODO turn off bias of batchnorm
        for bnlist in batchnormlist:
            for bnbn in bnlist:
                bnbn.bias = None


        # Initialize the firing thresholds of all the layers
        for m in self.modules():
            if (isinstance(m, nn.Conv2d)):
                m.threshold = 1.0
                torch.nn.init.xavier_normal_(m.weight,gain=2)
            elif (isinstance(m, nn.AvgPool2d)):
                m.threshold = 0.75
            elif (isinstance(m, nn.Linear)):
                m.threshold = 1.0
                torch.nn.init.xavier_normal_(m.weight,gain=2)


        # Instantiate differentiable spiking nonlinearity
        self.spike_fn = init_spike_fn(self.grad_type)
        self.spike_pool = init_spike_fn(self.grad_type_pool)

    def fc_init(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)

        torch.nn.init.xavier_uniform_(self.fc2.weight)


    def forward(self, inp):


        # avg_spike_time = []
        # Initialize the neuronal membrane potentials and dropout masks
        batch_size = inp.size(0)
        mem_conv1 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv2 = torch.zeros(batch_size, 128, self.img_size//2, self.img_size//2).cuda()
        mem_conv3 = torch.zeros(batch_size, 256, self.img_size//4, self.img_size//4).cuda()
        mem_conv4 = torch.zeros(batch_size, 256, self.img_size//4, self.img_size//4).cuda()
        mem_conv5 = torch.zeros(batch_size, 512, self.img_size//8, self.img_size//8).cuda()
        mem_conv6 = torch.zeros(batch_size, 512, self.img_size//8, self.img_size//8).cuda()
        mem_conv7 = torch.zeros(batch_size, 512, self.img_size // 16, self.img_size // 16).cuda()
        mem_conv8 = torch.zeros(batch_size, 512, self.img_size // 16, self.img_size // 16).cuda()


        mem_fc1 = torch.zeros(batch_size, 4*1024).cuda()
        mem_fc2 = torch.zeros(batch_size, self.num_cls).cuda()

        fc_dropout_mask = self.drop(torch.ones([batch_size, 1024]).cuda())

        for t in range(self.num_steps):
            spike_inp = PoissonGen(inp)
            # spike_x = torch.cat([spike_inp]*22,1)[:,:64,:,:]
            out_prev = spike_inp

            # Compute the conv1 outputs
            mem_thr   = (mem_conv1/self.conv1.threshold) - 1.0
            out       = self.spike_fn(mem_thr)
            rst       = torch.zeros_like(mem_conv1).cuda()
            rst[mem_thr>0] = self.conv1.threshold
            mem_conv1 = (self.leak_mem*mem_conv1 + self.bn1_list[int(t/self.one_stamp)](self.conv1(out_prev)) -rst)
            out_prev  = out.clone()

            # Compute the avgpool1 outputs
            out =  self.pool1(out_prev)
            out_prev = out.clone()


            # Compute the conv2 outputs
            mem_thr   = (mem_conv2/self.conv2.threshold) - 1.0
            out       = self.spike_fn(mem_thr)
            rst       = torch.zeros_like(mem_conv2).cuda()
            rst[mem_thr>0] = self.conv2.threshold
            mem_conv2 = (self.leak_mem*mem_conv2 + self.bn2_list[int(t/self.one_stamp)](self.conv2(out_prev)) -rst)
            out_prev  = out.clone()

            # Compute the avgpool2 outputs
            out = self.pool2(out_prev)
            out_prev = out.clone()



            # Compute the conv3 outputs
            mem_thr = (mem_conv3 / self.conv3.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv3).cuda()
            rst[mem_thr > 0] = self.conv3.threshold
            mem_conv3 = (self.leak_mem * mem_conv3 + self.bn3_list[int(t/self.one_stamp)](self.conv3(out_prev)) - rst)
            out_prev = out.clone()

            # Compute the conv4 outputs
            mem_thr = (mem_conv4 / self.conv4.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv4).cuda()
            rst[mem_thr > 0] = self.conv4.threshold
            mem_conv4 = (self.leak_mem * mem_conv4 + self.bn4_list[int(t/self.one_stamp)](self.conv4(out_prev)) - rst)
            out_prev = out.clone()

            # Compute the avgpool3 outputs
            out = self.pool3(out_prev)
            out_prev = out.clone()



            # Compute the conv5 outputs
            mem_thr = (mem_conv5 / self.conv5.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv5).cuda()
            rst[mem_thr > 0] = self.conv5.threshold
            mem_conv5 = (self.leak_mem * mem_conv5 + self.bn5_list[int(t/self.one_stamp)](self.conv5(out_prev)) - rst)
            out_prev = out.clone()

            # Compute the conv6 outputs
            mem_thr = (mem_conv6 / self.conv6.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv6).cuda()
            rst[mem_thr > 0] = self.conv6.threshold
            mem_conv6 = (self.leak_mem * mem_conv6 + self.bn6_list[int(t/self.one_stamp)](self.conv6(out_prev)) - rst)
            out_prev = out.clone()

            # Compute the avgpool4 outputs
            out = self.pool4(out_prev)
            out_prev = out.clone()



            # Compute the conv7 outputs
            mem_thr = (mem_conv7 / self.conv7.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv7).cuda()
            rst[mem_thr > 0] = self.conv7.threshold
            mem_conv7 = (self.leak_mem * mem_conv7 + self.bn7_list[int(t / self.one_stamp)](self.conv7(out_prev)) - rst)
            out_prev = out.clone()

            # Compute the conv8 outputs
            mem_thr = (mem_conv8 / self.conv8.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv8).cuda()
            rst[mem_thr > 0] = self.conv8.threshold
            mem_conv8 = (self.leak_mem * mem_conv8 + self.bn8_list[int(t / self.one_stamp)](self.conv8(out_prev)) - rst)
            out_prev = out.clone()

            # Compute the avgpool5 outputs
            out = self.avg_pool(out_prev)
            out_prev = out.clone()




            out_prev = out_prev.reshape(batch_size, -1)
            

            # compute fc1
            mem_thr = (mem_fc1 / self.fc1.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_fc1).cuda()
            rst[mem_thr > 0] = self.fc1.threshold
            mem_fc1 = (self.leak_mem * mem_fc1 + self.bnfc_list[int(t/self.one_stamp)](self.fc1(out_prev)) - rst)

            out_prev = out.clone()

            # compute fc1
            mem_fc2 = (1 * mem_fc2 + self.fc2(out_prev))


        out_voltage  = mem_fc2
        out_voltage = (out_voltage) / self.num_steps


        return out_voltage
# ----------------------------------------------


