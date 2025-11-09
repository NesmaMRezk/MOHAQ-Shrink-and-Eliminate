import copy
import warnings
import math
from typing import List, Tuple, Union, Optional

import torch
import torch.nn as nn
from pandas.tests.io.excel.test_xlsxwriter import xlsxwriter
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
#from quantization.q_utils import linear_quantize_clamp, get_tensor_max_abs, symmetric_linear_quantization_scale_factor
def quantize_16_act(vector):
    # maximum is less than 62.xxxx
    # 1 bit for sign, 5 bit for whole, 10 bit for fraction part
    #print(get_tensor_max_abs(vector))
    if get_tensor_max_abs(vector)<31:
        quantized_vector = torch.round(vector * 1000) / 1000
    else:
        quantized_vector = torch.round(vector * 100) / 100
    return quantized_vector

def symmetric_linear_quantization_scale_factor(num_bits, saturation_val):
    # Leave one bit for sign
    n = 2 ** (num_bits - 1) - 1
    return n / saturation_val

def get_tensor_max_abs(tensor):
    return max(abs(tensor.max().item()), abs(tensor.min().item()))


def asymmetric_linear_quantization_scale_factor(num_bits, saturation_min, saturation_max):
    n = 2 ** num_bits - 1
    return n / (saturation_max - saturation_min)


def clamp(input, min, max, inplace=False):
    if inplace:
        input.clamp_(min, max)
        return input
    return torch.clamp(input, min, max)


def linear_quantize(input, scale_factor, inplace=False):
    if inplace:
        input.mul_(scale_factor).round_()
        return input
    return torch.round(scale_factor * input)


def linear_quantize_clamp(input, scale_factor, clamp_min, clamp_max, inplace=False):
    input_o=input
    output = linear_quantize(input, scale_factor, inplace)

    return clamp(output, clamp_min, clamp_max, inplace)




def quantize_fixed16(vector):
    #for idx, input in enumerate(vector):
    #    temp=[]
        # Quantize input into 16 bit fixed point: 1-bit sign 4-bit real part 11-bit fraction
    #    for i, value in enumerate(input[0]):
            # print(value)
    #        vector[idx][i].data=Decimal(value.item()).quantize(Decimal('.001'), rounding=ROUND_DOWN)
        #vector[idx].data=temp
        # print (value)
    #TODO generalize
    vector=torch.round(vector*1000)/1000

    return vector


def requantize(x,accum_max_abs,number_of_bits,input_scale,w_scale,min,max,flag,type='16',stat=False,Testcase=0,scale_16=1,k=100):


    #if type== 'wf' :
     #   print(type+ " max " + str(get_tensor_max_abs(x)))
    if not flag:
        return x

    if Testcase==6 and (type=='ft' or type=='it' or type=='ot' or type=='ct'):
        x=x*scale_16/accum_max_abs *0.6 # : 0.6 = is for the 2 layer model, compute another value for 3 layers model (1.2)
        #print(type + " max " + str(get_tensor_max_abs(x)))
        #return quantize_fixed16(x)



    if number_of_bits==16:
        return quantize_fixed16(x)

    keep_threshold=True
    if Testcase==2 :
        keep_threshold=False
    if type == 'MLP':
        keep_threshold=True

    accum_scale=input_scale*w_scale
    if Testcase==3:
        accum_scale = 1

        if (type=='wf' or type=='wi' or type=='wo' or type=='wc' or type=='ft' or type=='it' or type=='ot' or type=='ct') :
            keep_threshold = False
        if (type == 'ft' or type == 'it' or type == 'ot' or type == 'ct') and number_of_bits == 8:
            number_of_bits = 7
    if (Testcase== 5 or Testcase==25 )and type =='x':
         accum_max_abs=1.01
    # for two layer model multiply by 1.01
    # three layer model precision8 multiply with 25 with 2.5 (4 bits)


    if Testcase == 25 and type == 'ht':
            #accum_max_abs = 1.5
            accum_max_abs = 1.6
        # for two layer model multiply by 1.5 wer=21.2, 1.75 in case using MMSE
        # three layer model precision8 multiply with 25 wer=20.5 with 4 (3 bits) wer=21.9
       # three layer 8 bit 1.25


   # clamp(x, min, max, inplace=True)
    if type=='x' and not (Testcase==4 or Testcase==5 or Testcase==25):
        return x
    """ re-quantize accumulator to quantized output range """
    if Testcase==1 or Testcase==2 or Testcase ==3 or type == 'MLP' or type == 'ht'or((Testcase==4 or Testcase==5 or Testcase==25) and type=='x') :
        requant_scale,out = post_quantized_forward(accum_max_abs,x,accum_scale,number_of_bits,stat,keep_threshold,type)

        x = linear_quantize_clamp(x, requant_scale, min, max, inplace=True)
       # print(type + str(get_tensor_max_abs(x)))

    return x
def duplicate_vector_small(vector,list,Bi=False):
    newlist = [None]*(len(list)-1)
    newlist2 = [None]*(len(list)-1)
    for i, v in enumerate(list):
        if v != '\n':
            if Bi == False:
                new = vector[int(v)]
                newlist[i]=new
            else:
                new = vector[0][int(v)]
                newlist[i]=new
                new2 = vector[1][int(v)]
                newlist2[i]=new2

    return newlist,newlist2


def duplicate_vector(vector,str_list,Bi=False):

    list=str_list.split(",")

  #  trial=vector[0][0]
   # trial1=vector[0][1]
    if Bi:
        if len(vector.shape)==3:
            new_vector = torch.rand(len(vector),2, len(vector[0][0]) + len(list) - 1, device='cuda')
        else:
            new_vector = torch.rand(len(vector), len(vector[0]) + len(list) - 1, device='cuda')

    else:
        new_vector = torch.rand(len(vector), len(vector[0]) + len(list) - 1, device='cuda')

    if len(vector.shape) == 3 or (len(vector.shape)==2 and not Bi) :
        for j in range(len(vector)):

            newlist,newlist2=duplicate_vector_small(vector[j],list,Bi)
            if not Bi:
                new_vector[j]=torch.cat([vector[j],torch.FloatTensor(newlist).cuda()])
            else:
                new_vector[j][0] = torch.cat([vector[j][0], torch.FloatTensor(newlist).cuda()])
                new_vector[j][1] = torch.cat([vector[j][1], torch.FloatTensor(newlist2).cuda()])
    else:
        newlist, newlist2 = duplicate_vector_small(vector, list, Bi)
        if Bi:
            new_vector[0]=torch.cat([vector[0],torch.FloatTensor(newlist).cuda()])
            new_vector[1] = torch.cat([vector[1], torch.FloatTensor(newlist2).cuda()])
        else:
            new_vector[0] = torch.cat([vector[0], torch.FloatTensor(newlist).cuda()])

    return new_vector
def post_quantized_forward(accum_max_abs,accumulator,current_accum_scale,num_bits_acts,stat=False,keep_threshold=False,type='16'):

        if not keep_threshold:# and num_bits_acts==8:
            if num_bits_acts==8 or num_bits_acts==7:
                accum_max_abs=accum_max_abs*2.5
                #  multiply by 3
                # 3 layer 8 bit multiply by 1.5

            elif num_bits_acts==4:
                accum_max_abs = accum_max_abs * 1.5


        if stat:
            accum_max_abs=get_tensor_max_abs(accumulator)


        y_f_max_abs = accum_max_abs / current_accum_scale # try to get float point max


        if y_f_max_abs==0:
            out_scale=0
        else:
            out_scale = symmetric_linear_quantization_scale_factor(num_bits_acts, y_f_max_abs) # the original scale should be
        requant_scale = out_scale / current_accum_scale  # the current scale after taking inputs scaling into consideration
        #print(requant_scale)
        return requant_scale, out_scale

from sru.ops import (elementwise_recurrence_cpu,
                     elementwise_recurrence_gpu,
                     elementwise_recurrence_naive)


class SRUCell(nn.Module):
    """
    A single SRU layer as per `LSTMCell`, `GRUCell` in Pytorch.
    """

    __constants__ = ['input_size', 'hidden_size', 'output_size', 'rnn_dropout',
                     'dropout', 'bidirectional', 'has_skip_term', 'highway_bias',
                     'v1', 'rescale', 'activation_type', 'activation', 'custom_m',
                     'projection_size', 'num_matrices', 'layer_norm', 'weight_proj',
                     'scale_x']

    scale_x: Tensor
    weight_proj: Optional[Tensor]

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 dropout: float = 0.0,
                 rnn_dropout: float = 0.0,
                 bidirectional: bool = False,
                 n_proj: int = 0,
                 use_tanh: bool = False,
                 highway_bias: float = 0.0,
                 has_skip_term: bool = True,
                 layer_norm: bool = False,
                 rescale: bool = True,
                 v1: bool = False,
                 custom_m: Optional[nn.Module] = None,
                 amp_recurrence_fp16: bool = False):
        """Initialize the SRUCell module.

        Parameters
        ----------
        input_size: int
            the number of features in the input `x`
        hidden_size: int
            the number of features in the hidden state *for each
            direction*
        dropout: float, optional
            the dropout value applied between layers (default=0)
        rnn_dropout: float, optional
            [DEPRECATED] the variational dropout value (default=0)
            This option is deprecated because minimal performance
            improvement, and increases codebase size. This option will
            be removed at the next major version upgrade
        bidirectional: bool, optional
            if True, set the module as a bidirectional SRU
            (default=False)
        n_proj: int, optional
            if non-zero, factorize the ``weight`` parameter matrix as a
            product of two parameter matrices, using an innder dimension
            ``n_proj`` (default=0)
        use_tanh: bool, optional
            [DEPRECATED] if True, apply `tanh` activation to the hidden
            state (default=False). `tanh` is deprecated because minimal
            performance improvement, and increases codebase size. This
            option will be removed at the next major version upgrade.
        highway_bias: float, optional
            the initial value of the bias used in the highway (sigmoid)
            gate (defulat=0)
        has_skip_term: bool, optional
            whether to include a residual connection for output hidden
            state `h` (default=True)
        layer_norm: bool, optional
            whether to apply pre- layer normalization for this layer
            (default=False)
        rescale: bool, optional
            whether to apply a constant rescaling multiplier for the
            residual term (default=True)
        v1: bool, optional
            [DEPRECATED] whether to use the an ealier v1 implementation
            of SRU (default=False)
        custom_m: nn.Module, optional
            use the give module instead of the batched matrix
            multiplication to compute the intermediate representations U
            needed for the elementwise recurrrence operation
            (default=None)
        amp_recurrence_fp16: Type, optional
            When using AMP autocast, selects which type to use
            for recurrence custom kernel.
            False: torch.float32, True: torch.float16
        """
        super(SRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size  # hidden size per direction
        self.output_size = hidden_size * 2 if bidirectional else hidden_size
        self.rnn_dropout = float(rnn_dropout)
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        self.has_skip_term = has_skip_term
        self.highway_bias = highway_bias
        self.v1 = v1
        self.rescale = rescale
        self.activation_type = 0
        self.activation = 'none'
        self.custom_m: Optional[nn.Module] = custom_m
        if use_tanh:
            self.activation_type = 1
            self.activation = 'tanh'
        self.amp_recurrence_fp16 = amp_recurrence_fp16

        # projection dimension
        self.projection_size = 0
        if n_proj > 0 and n_proj < self.input_size and n_proj < self.output_size:
            self.projection_size = n_proj

        # number of sub-matrices used in SRU
        self.num_matrices = 3
        if has_skip_term and self.input_size != self.output_size:
            self.num_matrices = 4

        # make parameters
        if self.custom_m is None:
            if self.projection_size == 0:
                self.weight_proj = None
                self.weight = nn.Parameter(torch.Tensor(
                    input_size,
                    self.output_size * self.num_matrices
                ))
            else:
                self.weight_proj = nn.Parameter(torch.Tensor(input_size, self.projection_size))
                self.weight = nn.Parameter(torch.Tensor(
                    self.projection_size,
                    self.output_size * self.num_matrices
                ))
        self.weight_c = nn.Parameter(torch.Tensor(2 * self.output_size))
        self.bias = nn.Parameter(torch.Tensor(2 * self.output_size))

        # scaling constant used in highway connections when rescale=True
        self.register_buffer('scale_x', torch.FloatTensor([0]))

        self.layer_norm: Optional[nn.Module]= None
        if layer_norm:
            self.layer_norm = nn.LayerNorm(self.input_size)

        self.reset_parameters()

    def reset_parameters(self):
        """Properly initialize the weights of SRU, following the same
        recipe as:
        Xavier init:
            http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        Kaiming init:
            https://arxiv.org/abs/1502.01852

        """
        # initialize bias and scaling constant
        self.bias.data.zero_()
        bias_val, output_size = self.highway_bias, self.output_size
        self.bias.data[output_size:].zero_().add_(bias_val)
        self.scale_x.data[0] = 1
        if self.rescale and self.has_skip_term:
            # scalar used to properly scale the highway output
            scale_val = (1 + math.exp(bias_val) * 2)**0.5
            self.scale_x.data[0] = scale_val

        if self.custom_m is None:
            # initialize weights such that E[w_ij]=0 and Var[w_ij]=1/d
            d = self.weight.size(0)
            val_range = (3.0 / d)**0.5
            self.weight.data.uniform_(-val_range, val_range)
            if self.projection_size > 0:
                val_range = (3.0 / self.weight_proj.size(0))**0.5
                self.weight_proj.data.uniform_(-val_range, val_range)

            # projection matrix as a tensor of size:
            #    (input_size, bidirection, hidden_size, num_matrices)
            w = self.weight.data.view(d, -1, self.hidden_size, self.num_matrices)

            # re-scale weights for dropout and normalized input for better gradient flow
            if self.dropout > 0:
                w[:, :, :, 0].mul_((1 - self.dropout)**0.5)
            if self.rnn_dropout > 0:
                w.mul_((1 - self.rnn_dropout)**0.5)

            # making weights smaller when layer norm is used. need more tests
            if self.layer_norm:
                w.mul_(0.1)
                # self.weight_c.data.mul_(0.25)

            # properly scale the highway output
            if self.rescale and self.has_skip_term and self.num_matrices == 4:
                scale_val = (1 + math.exp(bias_val) * 2)**0.5
                w[:, :, :, 3].mul_(scale_val)
        else:
            if hasattr(self.custom_m, 'reset_parameters'):
                self.custom_m.reset_parameters()
            else:
                warnings.warn("Unable to reset parameters for custom module. "
                              "reset_parameters() method not found for custom module.")

        if not self.v1:
            # intialize weight_c such that E[w]=0 and Var[w]=1
            self.weight_c.data.uniform_(-3.0**0.5, 3.0**0.5)

            # rescale weight_c and the weight of sigmoid gates with a factor of sqrt(0.5)
            if self.custom_m is None:
                w[:, :, :, 1].mul_(0.5**0.5)
                w[:, :, :, 2].mul_(0.5**0.5)
            self.weight_c.data.mul_(0.5**0.5)
        else:
            self.weight_c.data.zero_()
            self.weight_c.requires_grad = False

    def forward(self,
                input: Tensor,
                c0: Optional[Tensor] = None,
                mask_pad: Optional[Tensor] = None, layer=0,weight_list=[],weight_act_list=[],proj_list=[],proj_act_list=[],global_counter=0,stat=None) -> Tuple[Tensor, Tensor]:
        """The forward method of the SRU layer.
        """

        if input.dim() != 2 and input.dim() != 3:
            raise ValueError("Input must be 2 or 3 dimensional")

        batch_size = input.size(-2)
        if c0 is None:
            c0 = torch.zeros(batch_size, self.output_size, dtype=input.dtype,
                             device=input.device)
        # apply layer norm before activation (i.e. before SRU computation)
        residual = input
        if self.layer_norm is not None:
            input = self.layer_norm(input)

        # apply dropout for multiplication
        if self.training and (self.rnn_dropout > 0):
            mask = self.get_dropout_mask_((batch_size, input.size(-1)), self.rnn_dropout)
            input = input * mask.expand_as(input)

        # get the scaling constant; scale_x is a scalar
        scale_val: Optional[Tensor] = None
        scale_val = self.scale_x if self.rescale else None

        # get dropout mask
        mask_c: Optional[Tensor] = None
        if self.training and (self.dropout > 0):
            mask_c = self.get_dropout_mask_((batch_size, self.output_size),
                                            self.dropout)

        # compute U, V
        #   U is (length, batch_size, output_size * num_matrices)
        #   V is (output_size*2,) or (length, batch_size, output_size * 2) if provided
      #  c0 = requantize(c0, get_tensor_max_abs(c0), 8, 1, 1, -128, 127, True, 'ht', False, 4)

        list92_good=[-1,3,9,140,0,160,0,220,600,1000,1200,1300,55]
        list92=[-1,3.88,9,170,0,160,0,240,470,1100,1200,1500,55]
        list82 = [0, 0, 2300, 1000]  # two layer 4*4 47.2%
        list9 = [0, 7000, 5000, 1600]  # three layer 4*4 23.6%
        list10=[0,0,0,2000000] #one layer 8*8 23.4%
        list11= [600,1000,1200,1300] # two layer 8*8 27%
        list12=[220,1200,1200,1200]
        list41=[0,0,6,1000,15,300,25,300,1,1110,1250,1444,55]
        list31=[0,0,6,9,15,8.5,26.4,14.8,10.6,1138,1228.5,1584,55]
        list12=[0,3.88,3.5,960,15,670 ,18,291,472.5,1119.5,1257.5,1438,55]
        list26=[0,3.8,6.2,9,15.7,8.7,27,14.57,	468.5,1108.5,1275.5,1616.5,55]
        list72=[0,3.88,6,7648.7,16.4,6425,25.94,11581,105687,270.5,295.5,366,55]
        default4=[9,18,19,43.68] # 4 layers 24.4, 3 layers 23.7%, 2 layers 43.9%, 1 layer 39.3%
        default8 = [9, 18, 19,300] # on one layer gives 19% on two layers 24.7% on 3 layers 39.6% on 4 layers 37.9% (300 not 200)
        list122=[0,0,6,9,16.6,8.8,26,16,10.7,269,289.5,358,55]
        list112=[0,3.88,6,641.6,16.8,597,26.7,971,8465.5,268.5,295.5,373,55]
        list111=[0,3.8,	6.2,657.8,16.8,610,28,967,468.5,267.5,297,378.5,55]
        quantize_list=[True,True,True,True]
        quantize=True
        Test=False

        quantize_proj_list=[False,False,False,False]
  #      if ( layer>-1):
  #          input = requantize(input, get_tensor_max_abs(input)*0.8, 4, 1, 1, -8, 7, True, 'ht', False, 4)

        stat_list=list111
        U, V = self.compute_UV(input, c0, mask_pad,layer,quantize_list,quantize_proj_list,quantize,weight_list,weight_act_list,proj_list,proj_act_list,global_counter,stat,stat_list,Test)
        excel_list = ['H','I', 'J', 'K']
        if quantize:
            #requantize to fp
            #TODO statistics (I took the maximum in part of training set and clip to 80%)
            if stat != None:
                stat.write(excel_list[layer] + str(global_counter), str(get_tensor_max_abs(U)))
            #  print(get_tensor_max_abs(U))
            #list82[layer]=list12[layer]*0.8/default4[layer]#case 101 , list11 case 92
            if Test:
                list82[layer] = stat_list[8+layer] * 0.8 / default4[layer]  # case 101
            else:
                list82[layer] = get_tensor_max_abs(U) * 0.8 / default4[layer]  # case 101
            #list82[layer] =get_tensor_max_abs(U)  / default4[layer]
            U=U/list82[layer]
            U=quantize_16_act(U)
        # apply elementwise recurrence to get hidden states h and c
       # print(layer)
        #if (layer ==3):


        #U = requantize(U, get_tensor_max_abs(U), 8, 1, 1, -128, 127, True, 'ht', False, 4)

        h, c = self.apply_recurrence(U, V, residual, c0, scale_val, mask_c, mask_pad)

    #    c = requantize(c, get_tensor_max_abs(c), 8, 1, 1, -128, 127, True, 'ht', False, 4)
   #     h = requantize(h, get_tensor_max_abs(h), 8, 1, 1, -128, 127, True, 'ht', False, 4)
        if (layer==3 and quantize):
         #  print(get_tensor_max_abs(h))
        # TODO statistics

         if stat!=None:
            stat.write('L' + str(global_counter), str(get_tensor_max_abs(h)))
         #  h = requantize(h,44*0.7, 8, 1, 1, -128, 127, True, 'ht', False, 4)
           #h = requantize(h, 55* 0.8, 8, 1, 1, -128, 127, True, 'ht', False, 4) #case101
         #  h = requantize(h,60  * 0.8, 8, 1, 1, -128, 127, True, 'ht', False, 4) #case 92
         if Test:
             h = requantize(h, stat_list[12] * 0.8, 8, 1, 1, -128, 127, True, 'ht', False, 4)
         else:
             h = requantize(h, get_tensor_max_abs(h)* 0.8, 8, 1, 1, -128, 127, True, 'ht', False, 4)
         #print(c)
        return h, c

    def apply_recurrence(self,
                         U: Tensor,
                         V: Tensor,
                         residual: Tensor,
                         c0: Tensor,
                         scale_val: Optional[Tensor],
                         mask_c: Optional[Tensor],
                         mask_pad: Optional[Tensor]) -> List[Tensor]:
        """
        Apply the elementwise recurrence computation on given input
        tensors

        """
        if self.bias.is_cuda:
            return elementwise_recurrence_gpu(U, residual, V, self.bias, c0,
                                              self.activation_type,
                                              self.hidden_size,
                                              self.bidirectional,
                                              self.has_skip_term,
                                              scale_val, mask_c, mask_pad,
                                              self.amp_recurrence_fp16)

        if not torch.jit.is_scripting():
            return elementwise_recurrence_naive(U, residual, V, self.bias, c0,
                                                self.activation_type,
                                                self.hidden_size,
                                                self.bidirectional,
                                                self.has_skip_term,
                                                scale_val, mask_c, mask_pad)
        else:
            return elementwise_recurrence_cpu(U, residual, V, self.bias, c0,
                                              self.activation_type,
                                              self.hidden_size,
                                              self.bidirectional,
                                              self.has_skip_term,
                                              scale_val, mask_c, mask_pad)

    def compute_UV(self,
                   input: Tensor,
                   c0: Optional[Tensor],
                   mask_pad: Optional[Tensor],layer,quantize_list=[False,False,False,False],quantize_proj_list=[False,False,False,False],quantize=False,weight_list=[],weight_act_list=[],proj_list=[],proj_act_list=[],global_counter=0,stat=None,stat_list=[],Test=True) -> Tuple[Tensor, Tensor]:
        """
        SRU performs grouped matrix multiplication to transform the
        input (length, batch_size, input_size) into a tensor U of size
        (length * batch_size, output_size * num_matrices).

        When a custom module `custom_m` is given, U will be computed by
        the given module. In addition, the module can return an
        additional tensor V (length, batch_size, output_size * 2) that
        will be added to the hidden-to-hidden coefficient terms in
        sigmoid gates, i.e., (V[t, b, d] + weight_c[d]) * c[t-1].

        """
        if self.custom_m is None:
            U = self.compute_U(input,layer,quantize_list,quantize_proj_list,quantize,weight_list,weight_act_list,proj_list,proj_act_list,global_counter,stat,stat_list,Test)
            V = self.weight_c
        else:
            ret = self.custom_m(input)
            if isinstance(ret, tuple) or isinstance(ret, list):
                if len(ret) > 2:
                    raise Exception("Custom module must return 1 or 2 tensors but got {}.".format(
                        len(ret)
                    ))
                U, V = ret[0], ret[1] + self.weight_c
            else:
                U, V = ret, self.weight_c

            if U.size(-1) != self.output_size * self.num_matrices:
                raise ValueError("U must have a last dimension of {} but got {}.".format(
                    self.output_size * self.num_matrices,
                    U.size(-1)
                ))
            if V.size(-1) != self.output_size * 2:
                raise ValueError("V must have a last dimension of {} but got {}.".format(
                    self.output_size * 2,
                    V.size(-1)
                ))
        return U, V

    def compute_U(self,
                  input: Tensor,layer,quantize_list=[False,False,False,False],quantize_proj_list=[False,False,False,False],quantize=False,weight_list=[],weight_act_list=[],proj_list=[],proj_act_list=[],global_counter=0,stat=None,stat_list=[],Test=True) -> Tensor:
        """
        SRU performs grouped matrix multiplication to transform the
        input (length, batch_size, input_size) into a tensor U of size
        (length * batch_size, output_size * num_matrices)
        """
        # collapse (length, batch_size) into one dimension if necessary

        x = input if input.dim() == 2 else input.contiguous().view(-1, self.input_size)
        weight_proj = self.weight_proj
        excel_index_proj=['X','B','D','F']
        excel_index_layer=['A','C','E','G']
        if weight_proj is not None:
            if stat != None:
                stat.write(excel_index_proj[layer] + str(global_counter), str(get_tensor_max_abs(x)))
            if quantize:
                if quantize_proj_list[layer]==True:
                    # TODO statistics
                    if Test:
                        print(stat_list[layer*2])
                        print(get_tensor_max_abs(x))
                        x = requantize(x, stat_list[layer*2] * 0.8, 4, 1, 1, -8, 7, True, 'ht',
                                       False, 4)
                    else:
                        x = requantize(x, get_tensor_max_abs(x) * 0.8, 4, 1, 1, -8, 7, True, 'ht',
                                             False, 4)

                else:
                   # if (get_tensor_max_abs(x) > 31):
                   #     raise ValueError("Activation 16-bit quantization error")
                    x=quantize_16_act(x)
                proj_act_list.append(get_tensor_max_abs(x))
                proj_list.append(get_tensor_max_abs(weight_proj))
               # print("Weight_proj " + str(get_tensor_max_abs(weight_proj)))
            x_projected = x.mm(weight_proj)
            if stat != None:
                stat.write(excel_index_layer[layer] + str(global_counter), str(get_tensor_max_abs(x_projected)))

            if quantize:
                if quantize_list[layer] == True:
                    # TODO statistics

                    list=[0,100,100,180]
                    list2=[0,140,160,220]#case92
                    #print(weight_proj)
                   # print(get_tensor_max_abs(x_projected))
                    #case 101 list[layer]*0.7
                    if Test:
                        print(stat_list[2*layer+1])
                        print(get_tensor_max_abs(x_projected))
                        x_projected = requantize(x_projected, stat_list[2*layer+1]*0.8, 4, 1, 1, -8, 7, True, 'ht', False, 4)
                    else:
                        x_projected = requantize(x_projected, get_tensor_max_abs(x_projected) * 0.8, 4, 1, 1, -8, 7,
                                                 True, 'ht', False, 4)

                else:
                   # if (get_tensor_max_abs(x_projected) > 31):
                    #    raise ValueError("Activation 16-bit quantization error")
                    x_projected=quantize_16_act(x_projected)
                weight_act_list.append(get_tensor_max_abs(x_projected))
                weight_list.append(get_tensor_max_abs(self.weight))
               # print("X_projected " + str(get_tensor_max_abs(x_projected)))
               # print("Weight " + str(get_tensor_max_abs(self.weight)))

            U = x_projected.mm(self.weight)


        else:
            if quantize:
                if quantize_list[layer] == True:
                    if stat != None:
                        stat.write(excel_index_layer[layer]+str(global_counter),str(get_tensor_max_abs(x)))
                    # TODO statistics
                    if Test:
                        max=stat_list[2*layer+1]
                    else:
                        max=get_tensor_max_abs(x)
                    if layer==-1:

                        x = requantize(x, max * 0.8, 8, 1, 1, -128, 127, True, 'ht', False, 4)

                    else:
                      #  print(get_tensor_max_abs(x))
                        #resume work here
                        x = requantize(x, max*0.8, 4, 1, 1, -8, 7, True, 'ht', False, 4)

                else:
                 #   if (get_tensor_max_abs(x) > 31):
                  #      raise ValueError("Activation 16-bit quantization error")
                    x = quantize_16_act(x)
            proj_act_list.append(0)
            proj_list.append(0)
            weight_act_list.append(get_tensor_max_abs(x))
            weight_list.append(get_tensor_max_abs(self.weight))
          #  print("X " + str(get_tensor_max_abs(x)))
          #  print("Weight " + str(get_tensor_max_abs(self.weight)))
            U = x.mm(self.weight)

        # print model precision


        return U

    def get_dropout_mask_(self,
                          size: Tuple[int, int],
                          p: float) -> Tensor:
        """
        Composes the dropout mask for the `SRUCell`.
        """
        b = self.bias.data
        return b.new_empty(size).bernoulli_(1 - p).div_(1 - p)

    def extra_repr(self):
        s = "{input_size}, {hidden_size}"
        if self.projection_size > 0:
            s += ", projection_size={projection_size}"
        if self.dropout > 0:
            s += ", dropout={dropout}"
        if self.rnn_dropout > 0:
            s += ", rnn_dropout={rnn_dropout}"
        if self.bidirectional:
            s += ", bidirectional={bidirectional}"
        if self.highway_bias != 0:
            s += ", highway_bias={highway_bias}"
        if self.activation_type != 0:
            s += ", activation={activation}"
        if self.v1:
            s += ", v1={v1}"
        s += ", rescale={rescale}"
        if not self.has_skip_term:
            s += ", has_skip_term={has_skip_term}"
        if self.layer_norm:
            s += ", layer_norm=True"
        if self.custom_m is not None:
            s += ",\n  custom_m=" + str(self.custom_m)
        return s.format(**self.__dict__)

    def __repr__(self):
        s = self.extra_repr()
        if len(s.split('\n')) == 1:
            return "{}({})".format(self.__class__.__name__, s)
        else:
            return "{}({}\n)".format(self.__class__.__name__, s)


class SRU(nn.Module):
    """
    Implementation of Simple Recurrent Unit (SRU)
    """

    __constants__ = ['input_size', 'hidden_size', 'output_size', 'num_layers',
                     'dropout', 'rnn_dropout', 'projection_size', 'rnn_lst',
                     'bidirectional', 'use_layer_norm', 'has_skip_term',
                     'num_directions', 'nn_rnn_compatible_return', 'input_to_hidden']

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 2,
                 dropout: float = 0.0,
                 rnn_dropout: float = 0.0,
                 bidirectional: bool = False,
                 projection_size: int = 0,
                 use_tanh: bool = False,
                 layer_norm: bool = False,
                 highway_bias: float = 0.0,
                 has_skip_term: bool = True,
                 rescale: bool = False,
                 v1: bool = False,
                 nn_rnn_compatible_return: bool = False,
                 custom_m: Optional[Union[nn.Module, List[nn.Module]]] = None,
                 proj_input_to_hidden_first: bool = False,
                 amp_recurrence_fp16: bool = False):
        """Initialize the SRU module.

        Parameters
        ----------
        input_size: int
            the number of features in the input `x`
        hidden_size: int
            the number of features in the hidden state *for each
            direction*
        num_layers: int
            the number of stacked SRU layers (default=2)
        dropout: float, optional
            the dropout value applied between layers (default=0)
        rnn_dropout: float, optional
            [DEPRECATED] the variational dropout value (default=0)
            This option is deprecated because minimal performance
            improvement, and increases codebase size. This option will
            be removed at the next major version upgrade
        bidirectional: bool, optional
            if True, set the module as a bidirectional SRU
            (default=False)
        projection_size: int, optional
            if non-zero, factorize the ``weight`` parameter in each
            layeras a product of two parameter matrices, using an innder
            dimension ``projection_size`` (default=0)
        use_tanh: bool, optional
            [DEPRECATED] if True, apply `tanh` activation to the hidden
            state (default=False). `tanh` is deprecated because minimal
            performance improvement, and increases codebase size. This
            option will be removed at the next major version upgrade.
        layer_norm: bool, optional
            whether to apply pre- layer normalization for this layer
            (default=False)
        highway_bias: float, optional
            the initial value of the bias used in the highway (sigmoid)
            gate (defulat=0)
        has_skip_term: bool, optional
            whether to include a residual connection for output hidden
            state `h` (default=True)
        rescale: bool, optional
            whether to apply a constant rescaling multiplier for the
            residual term (default=False)
        v1: bool, optional
            [DEPRECATED] whether to use the an ealier v1 implementation
            of SRU (default=False)
        custom_m: Union[nn.Module, List[nn.Module]], optional
            use the given module(s) instead of the batched matrix
            multiplication to compute the intermediate representations U
            needed for the elementwise recurrrence operation.  The
            module must take input x of shape (seq_len, batch_size,
            hidden_size). It returns a tensor U of shape (seq_len,
            batch_size, hidden_size * num_matrices), and one optional
            tensor V of shape (seq_len, batch_size, hidden_size * 2).
            (default=None)
        amp_recurrence_fp16: Type, optional
            When using AMP autocast, selects which type to use
            for recurrence custom kernel.
            False: torch.float32, True: torch.float16

        """

        super(SRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = hidden_size * 2 if bidirectional else hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn_dropout = rnn_dropout
        self.projection_size = projection_size
        self.bidirectional = bidirectional
        self.use_layer_norm = layer_norm
        self.has_skip_term = has_skip_term
        self.num_directions = 2 if bidirectional else 1
        self.nn_rnn_compatible_return = nn_rnn_compatible_return
        self.input_to_hidden = None
        self.excel=xlsxwriter.Workbook('range_statistics.xlsx')
        self.stat=self.excel.add_worksheet()
        self.stat.write('A1','ip-to-lay0')
        self.stat.write('B1', 'ip-to-proj1')
        self.stat.write('C1', 'ip-to-lay1')
        self.stat.write('D1', 'ip-to-proj2')
        self.stat.write('E1', 'ip-to-lay2')
        self.stat.write('F1', 'ip-to-proj3')
        self.stat.write('G1', 'ip-to-lay3')
        self.stat.write('H1','MxV_0')
        self.stat.write('I1', 'MxV_1')
        self.stat.write('J1', 'MxV_2')
        self.stat.write('K1', 'MxV_3')
        self.stat.write('L1', 'SRU_output')
        self.global_counter=2
       # self.amp_recurrence_fp16=True
        print("fp")
        print(amp_recurrence_fp16)

        if proj_input_to_hidden_first and input_size != self.output_size:
            first_layer_input_size = self.output_size
            self.input_to_hidden = nn.Linear(input_size, self.output_size, bias=False)
        else:
            first_layer_input_size = input_size
        self.amp_recurrence_fp16 = amp_recurrence_fp16

        if rnn_dropout > 0:
            warnings.warn("rnn_dropout > 0 is deprecated and will be removed in"
                          "next major version of SRU. Please use dropout instead.")
        if use_tanh:
            warnings.warn("use_tanh = True is deprecated and will be removed in"
                          "next major version of SRU.")

        rnn_lst = nn.ModuleList()
        for i in range(num_layers):
            # get custom modules when provided
            custom_m_i = None
            if custom_m is not None:
                custom_m_i = custom_m[i] if isinstance(custom_m, list) else copy.deepcopy(custom_m)
            # create the i-th SRU layer
            layer_i = SRUCell(
                first_layer_input_size if i == 0 else self.output_size,
                self.hidden_size,
                dropout=dropout if i + 1 != num_layers else 0,
                rnn_dropout=rnn_dropout,
                bidirectional=bidirectional,
                n_proj=projection_size,
                use_tanh=use_tanh,
                layer_norm=layer_norm,
                highway_bias=highway_bias,
                has_skip_term=has_skip_term,
                rescale=rescale,
                v1=v1,
                custom_m=custom_m_i,
                amp_recurrence_fp16=amp_recurrence_fp16
            )
            rnn_lst.append(layer_i)
        self.rnn_lst = rnn_lst

    def forward(self, input: Tensor,
                c0: Optional[Tensor] = None,
                mask_pad: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """The forward method of SRU module

        Parameters
        ----------
        input: Tensor
            the input feature. shape: (length, batch_size, input_size)
        c0: Tensor, optional
            the initial internal hidden state. shape: (num_layers,
            batch_size, output_size) where
            output_size = hidden_size * num_direction
        mask_pad: Tensor, optional
            the mask where a non-zero value indicates if an input token
            is pad token that should be ignored in forward and backward
            computation. shape: (length, batch_size)

        Returns
        ----------
        h: Tensor
            the output hidden state. shape: (length, batch_size,
            output_size) where
            output_size = hidden_size * num_direction
        c: Tensor
            the last internal hidden state. shape: (num_layers,
            batch_size, output_size), or (num_layers * num_directions,
            batch_size, hidden_size) if `nn_rnn_compatible_return` is
            set `True`

        """
        # unpack packed, if input is packed. packing and then unpacking will be slower than not
        # packing at all, but makes SRU usage compatible with nn.RNN usage
        orig_input = input
        if isinstance(orig_input, PackedSequence):
            input, lengths = nn.utils.rnn.pad_packed_sequence(input)
            max_length = lengths.max().item()
            mask_pad = torch.ByteTensor([[0] * length + [1] * (max_length - length)
                                        for length in lengths.tolist()])
            mask_pad = mask_pad.to(input.device).transpose(0, 1).contiguous()

        # The dimensions of `input` should be: `(sequence_length, batch_size, input_size)`.
        if input.dim() != 3:
            raise ValueError("There must be 3 dimensions for (length, batch_size, input_size)")

        if c0 is None:
            zeros = torch.zeros(input.size(1), self.output_size, dtype=input.dtype,
                                device=input.device)
            c0_ = [zeros for i in range(self.num_layers)]
        else:
            # The dimensions of `c0` should be: `(num_layers, batch_size, hidden_size * dir_)`.
            if c0.dim() != 3:
                raise ValueError("c0 must be 3 dim (num_layers, batch_size, output_size)")
            c0_ = [x.squeeze(0) for x in c0.chunk(self.num_layers, 0)]

        if self.input_to_hidden is None:
            prevx = input
        else:
            prevx = self.input_to_hidden(input)
        lstc = []
        weight_list=[]
        weight_act_list=[]
        proj_list=[]
        proj_act_list=[]
        i = 0
        for rnn in self.rnn_lst:
        #    if i>0:
        #        prevx=requantize(prevx, get_tensor_max_abs(prevx),8, 1, 1,-128, 127,True,'ht',False,4)
          #      c0_[i] = requantize(c0_[i], get_tensor_max_abs(c0_[i]), 8, 1, 1, -128, 127, True, 'ht', False, 4)
            # print(prevx)
            #print(i)
            h, c = rnn(prevx, c0_[i], mask_pad=mask_pad,layer=i,weight_list=weight_list,weight_act_list=weight_act_list,proj_list=proj_list,proj_act_list=proj_act_list,global_counter=self.global_counter,stat=self.stat)
            prevx = h
            lstc.append(c)
            i += 1

        self.global_counter+=1
        print("\nP0 "+ str(proj_list[0]) + "*" +str(proj_act_list[0])+" \nW0 "+ str(weight_list[0]) + "*" +str(weight_act_list[0]))
        print( "P1 "+ str(proj_list[1]) + "*" +str(proj_act_list[1])+" \nW1 " + str(weight_list[1]) + "*" +str(weight_act_list[1]))
        print("P2 "+ str(proj_list[2]) + "*" +str(proj_act_list[2])+" \nW2 " + str(weight_list[2]) + "*" +str(weight_act_list[2]))
        print("P3 "+ str(proj_list[3]) + "*" +str(proj_act_list[3])+" \nW3 "+ str(weight_list[3]) + "*" +str(weight_act_list[3]) )

        if self.global_counter==50:
            self.excel.close()
            self.excel=None
            self.stat=None

        lstc_stack = torch.stack(lstc)
        if self.nn_rnn_compatible_return:
            batch_size = input.size(1)
            lstc_stack = lstc_stack.view(self.num_layers, batch_size,
                                         self.num_directions, self.hidden_size)
            lstc_stack = lstc_stack.transpose(1, 2).contiguous()
            lstc_stack = lstc_stack.view(self.num_layers * self.num_directions,
                                         batch_size, self.hidden_size)

        if isinstance(orig_input, PackedSequence):
            prevx = nn.utils.rnn.pack_padded_sequence(prevx, lengths, enforce_sorted=False)
            return prevx, lstc_stack
        else:
            return prevx, lstc_stack

    def reset_parameters(self):
        for rnn in self.rnn_lst:
            rnn.reset_parameters()
        if self.input_to_hidden is not None:
            self.input_to_hidden.reset_parameters()

    def make_backward_compatible(self):
        self.nn_rnn_compatible_return = getattr(self, 'nn_rnn_compatible_return', False)

        # version <= 2.1.7
        if hasattr(self, 'n_in'):
            if len(self.ln_lst):
                raise Exception("Layer norm is not backward compatible for sru<=2.1.7")
            if self.use_weight_norm:
                raise Exception("Weight norm removed in sru>=2.1.9")
            self.input_size = self.n_in
            self.hidden_size = self.n_out
            self.output_size = self.out_size
            self.num_layers = self.depth
            self.projection_size = self.n_proj
            self.use_layer_norm = False
            for cell in self.rnn_lst:
                cell.input_size = cell.n_in
                cell.hidden_size = cell.n_out
                cell.output_size = cell.n_out * 2 if cell.bidirectional else cell.n_out
                cell.num_matrices = cell.k
                cell.projection_size = cell.n_proj
                cell.layer_norm = None
                if cell.activation_type > 1:
                    raise Exception("ReLU or SeLU activation removed in sru>=2.1.9")

        # version <= 2.1.9
        if not hasattr(self, 'input_to_hidden'):
            self.input_to_hidden = None
            for cell in self.rnn_lst:
                cell.custom_m = None

