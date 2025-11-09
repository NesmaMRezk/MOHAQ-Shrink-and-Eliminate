import torch
import torch.nn as nn
import numpy as np

import sru
from .quantizer import Quantizer
from .q_utils import *
from .range_linear import RangeLinearQuantWrapper as RangeLinearActQuantWrapper
from .ocs_impl import ocs_wts
from .clip import find_clip_aciq, find_clip_mmse, find_clip_entropy


#Nesma: Updated the file to work with FC, LSTM, and SRU not only convolutions

# For activations profiling, we do a hacky implementation where setting
# a global var puts each instance of OCSParamLayerWrapper into profiling
# mode. In this mode we collect activation stats and don't perform
# quantization or OCS.torch.set_printoptions(threshold=10_000)
PROFILE_MODE = False

def ocs_set_profile_mode(pm):
    global PROFILE_MODE
    PROFILE_MODE = pm

#Nesma: old, not used
def compute_MSE(original,quantized):


    quan=quantized.flatten().copy()
    n=len(original)
    sum=0
    hq,nn = np.histogram(quan,n=256, density=True)
    hx,edges=np.histogram(original,n=256,density=True)
    diff=hq-hx
    error= (original - quan)**2
    error=np.multiply(torch.from_numpy(hx),error)
    error = torch.sum(error)
    error =error/n


    return error
class OCSParamLayerWrapper(RangeLinearActQuantWrapper):
    """
    OCS quantization wrappers for layers with weights (namely torch.nn.ConvNd and
    torch.nn.Linear)

    Args:
        wrapped_module (torch.nn.Module): Module to be wrapped
        num_bits_acts (int): Number of bits used for inputs and output quantization
        num_bits_params (int): Number of bits used for parameters (weights, no bias) quantization
        num_bits_accum (int): Number of bits allocated for the accumulator of intermediate integer results
    """
    def __init__(self, wrapped_module, num_bits_acts, num_bits_params, num_bits_accum=32,
                 weight_expand_ratio=0.0, weight_clip_threshold=1.0,
                 act_expand_ratio=0.0, act_clip_threshold=1.0,skip_layer_one=False, layer_one_size=69):
        super(OCSParamLayerWrapper, self).__init__(wrapped_module,
                                                        num_bits_acts, num_bits_accum)

        if not isinstance(wrapped_module, (nn.Conv2d, nn.Linear,sru.SRUCell)):
            raise ValueError(self.__class__.__name__ +
                             ' can wrap only Conv2D an_wtsd Linear modules')

        self.num_bits_params = num_bits_params.pop(0)
        if isinstance(wrapped_module,sru.SRUCell):
            self.num_bits_proj=num_bits_params.pop(0)
            self.params_min_q_val_proj, self.params_max_q_val_proj=get_quantized_range(self.num_bits_proj, signed=True)
        self.num_bits_acts = num_bits_acts
        self.params_min_q_val, self.params_max_q_val = get_quantized_range(self.num_bits_params, signed=True)

        self.weight_expand_ratio = weight_expand_ratio
        self.weight_clip_threshold = weight_clip_threshold
        self.act_expand_ratio = act_expand_ratio
        self.act_clip_threshold = act_clip_threshold
        self.split_threshold = 0.0

        self.current_accum_scale = 1

        # Profiling
        self.profile_info = None
        self.weight_orig = None

        self.channels_to_split = None
    #Nesma: I added similar pieces of code for linear and SRUCell
    #Nesma: Allowed simple binary quantization for SRU
        if isinstance(wrapped_module, nn.Conv2d):
            weight_torch = wrapped_module.weight.data
            self.weight_orig = weight_torch
            weight_np = weight_torch.numpy()

            """ Avoid quantizing the input layer """
            num_channels = weight_torch.shape[1]
            if num_channels == 3:
                return

            # Perform prelim OCS
            q_weight_np, in_channels_to_split,in_channels_to_split_str = ocs_wts(
                    weight_np,
                    self.weight_expand_ratio,
                    split_threshold=self.split_threshold,
                    grid_aware=False)

            # Find the clip threshold (alpha value in clipping papers)
            if self.weight_clip_threshold > 0.0:
                # Fixed threshold
                max_abs = get_tensor_max_abs(q_weight_np)
                clip_max_abs = self.weight_clip_threshold * max_abs
            else:
           #     print('Auto-tuning for weight clip threshold...')
                # Calculate threshold
                values = weight_np.flatten().copy()

                # Branch on clip method
                if self.weight_clip_threshold == 0.0:
                    clip_max_abs = find_clip_mmse(values, self.num_bits_params)
                elif self.weight_clip_threshold == -1.0:
                    clip_max_abs = find_clip_aciq(values, self.num_bits_params)
                elif self.weight_clip_threshold == -2.0:
                    clip_max_abs = find_clip_entropy(values, self.num_bits_params)
                else:
                    raise ValueError('Undefined weight clip method')


            self.w_scale = symmetric_linear_quantization_scale_factor(self.num_bits_params, clip_max_abs)

            # Grid aware OCS
            q_weight_np, in_channels_to_split,in_channels_to_split_str = ocs_wts(
                    weight_np,
                    self.weight_expand_ratio,
                    split_threshold=self.split_threshold,
                    w_scale=self.w_scale,
                    grid_aware=True)

            # Save which channels got split
            if len(in_channels_to_split) > 0:
                assert(q_weight_np.shape[1] == num_channels + len(in_channels_to_split))
                self.channels_to_split = np.array(in_channels_to_split)
                self.channels_to_split = torch.from_numpy(self.channels_to_split).cuda()

            q_weight_torch = torch.from_numpy(q_weight_np)
            linear_quantize_clamp(q_weight_torch,
                                  self.w_scale,
                                  self.params_min_q_val,
                                  self.params_max_q_val,
                                  inplace=True)
            wrapped_module.weight.data = q_weight_torch
        if isinstance(wrapped_module, nn.Linear):
            weight_torch = wrapped_module.weight.data
            if wrapped_module.bias is not None:
                bias_torch=wrapped_module.bias.data
                bias_np = bias_torch.numpy()

            self.weight_orig = weight_torch
            weight_np = weight_torch.numpy()


            #""" Avoid quantizing the input layer """

            num_channels = weight_torch.shape[1]
            if (num_channels == layer_one_size and skip_layer_one):
                return

            # Perform prelim OCS
            q_weight_np, in_channels_to_split,in_channels_to_split_str = ocs_wts(
                    weight_np,
                    self.weight_expand_ratio,
                    split_threshold=self.split_threshold,
                    grid_aware=False,Convolution=False)

            # Find the clip threshold (alpha value in clipping papers)
            if self.weight_clip_threshold > 0.0:
                # Fixed threshold
                max_abs = get_tensor_max_abs(q_weight_np)
                clip_max_abs = self.weight_clip_threshold * max_abs
                clip_bias_max_abs = self.weight_clip_threshold * max_abs
            else:
             #   print('Auto-tuning for weight clip threshold...')
                # Calculate threshold
                values = weight_np.flatten().copy()
                if wrapped_module.bias is not None:
                    values_b=bias_np.flatten().copy()

                # Branch on clip method
                if self.weight_clip_threshold == 0.0:
                    clip_max_abs = find_clip_mmse(values, self.num_bits_params)
                    if wrapped_module.bias is not None:
                        clip_bias_max_abs = find_clip_mmse(values_b, self.num_bits_params)
                elif self.weight_clip_threshold == -1.0:
                    clip_max_abs = find_clip_aciq(values, self.num_bits_params)
                    if wrapped_module.bias is not None:
                        clip_bias_max_abs = find_clip_aciq(values_b, self.num_bits_params)
                elif self.weight_clip_threshold == -2.0:
                    clip_max_abs = find_clip_entropy(values, self.num_bits_params)
                    if wrapped_module.bias is not None:
                        clip_bias_max_abs = find_clip_entropy(values_b, self.num_bits_params)
                else:
                    raise ValueError('Undefined weight clip method')

            self.w_scale = symmetric_linear_quantization_scale_factor(self.num_bits_params, clip_max_abs)
            if wrapped_module.bias is not None:
                self.b_scale = symmetric_linear_quantization_scale_factor(self.num_bits_params, clip_bias_max_abs)

            # Grid aware OCS
            q_weight_np, in_channels_to_split,in_channels_to_split_str = ocs_wts(
                    weight_np,
                    self.weight_expand_ratio,
                    split_threshold=self.split_threshold,
                    w_scale=self.w_scale,
                    grid_aware=True,Convolution=False)

            # Save which channels got split
            if len(in_channels_to_split) > 0:
                assert(q_weight_np.shape[1] == num_channels + len(in_channels_to_split))
                self.channels_to_split = np.array(in_channels_to_split)
                self.channels_to_split = torch.from_numpy(self.channels_to_split).cuda()

            q_weight_torch = torch.from_numpy(q_weight_np)
            q_weight_torch_org=q_weight_torch.clone()
            if wrapped_module.bias is not None:
                q_bias_torch = wrapped_module.bias.data
            linear_quantize_clamp(q_weight_torch,
                                  self.w_scale,
                                  self.params_min_q_val,
                                  self.params_max_q_val,
                                  inplace=True)
            if wrapped_module.bias is not None:
                linear_quantize_clamp(q_bias_torch,
                                  self.b_scale,
                                  self.params_min_q_val,
                                  self.params_max_q_val,
                                  inplace=True)
        #    print(compute_MSE(values,q_weight_np))
            wrapped_module.weight.data = q_weight_torch

            wrapped_module.split=in_channels_to_split_str

            wrapped_module.weight
        if isinstance(wrapped_module, sru.SRUCell):
            if self.num_bits_params==1:
                binary_flag=True
            else:
                binary_flag=False
            if self.num_bits_proj==1:
                binary_proj_flag=True
            else:
                binary_proj_flag=False
            weight_torch = wrapped_module.weight.data
            weight_c_torch=wrapped_module.weight_c.data
            if wrapped_module.projection_size is not 0:
                weight_proj_torch=wrapped_module.weight_proj.data
            if wrapped_module.bias is not None:
                bias_torch=wrapped_module.bias.data
                bias_np = bias_torch.numpy()

            self.weight_orig = weight_torch
            weight_np = weight_torch.numpy()
            self.weight_c_orig = weight_c_torch
            weight_c_np = weight_c_torch.numpy()
            if wrapped_module.projection_size is not 0:
                self.weight_proj_orig = weight_proj_torch
                weight_proj_np = weight_proj_torch.numpy()


            #""" Avoid quantizing the input layer """

            num_channels = weight_torch.shape[1]
            num_channels_c = weight_c_torch.shape[0]
            if wrapped_module.projection_size is not 0:
                num_channels_proj = weight_proj_torch.shape[0]
            if (num_channels == layer_one_size and skip_layer_one):
                return

           # for i, v in enumerate(wrapped_module.weight.data):
           #     if i<4:
           #         print(i,v)

            # Perform prelim OCS
            q_weight_np, in_channels_to_split,in_channels_to_split_str = ocs_wts(
                    weight_np,
                    self.weight_expand_ratio,
                    split_threshold=self.split_threshold,
                    grid_aware=False,Convolution=False)
            q_weight_c_np, in_channels_to_split_c,in_channels_to_split_str_c = ocs_wts(
                    weight_c_np,
                    self.weight_expand_ratio,
                    axis=0,
                    split_threshold=self.split_threshold,
                    grid_aware=False,Convolution=False)
            if wrapped_module.projection_size is not 0:
                q_weight_proj_np, in_channels_to_split_proj,in_channels_to_split_str_proj = ocs_wts(
                    weight_proj_np,
                    self.weight_expand_ratio,
                    axis=0,
                    split_threshold=self.split_threshold,
                    grid_aware=False,Convolution=False)

            # Find the clip threshold (alpha value in clipping papers)
            if self.weight_clip_threshold > 0.0:
                # Fixed threshold
                max_abs = get_tensor_max_abs(q_weight_np)
                clip_max_abs = self.weight_clip_threshold * max_abs
                clip_bias_max_abs = self.weight_clip_threshold * max_abs
                max_abs_c = get_tensor_max_abs(q_weight_c_np)
                clip_max_abs_c = self.weight_clip_threshold * max_abs_c
                if wrapped_module.projection_size is not 0:
                    max_abs_proj = get_tensor_max_abs(q_weight_proj_np)
                    clip_max_abs_proj = self.weight_clip_threshold * max_abs_proj

            else:
             #   print('Auto-tuning for weight clip threshold...')
                # Calculate threshold
                values = weight_np.flatten().copy()
                values_c = weight_c_np.flatten().copy()
                if wrapped_module.projection_size is not 0:
                    values_proj = weight_proj_np.flatten().copy()
                if wrapped_module.bias is not None:
                    values_b=bias_np.flatten().copy()

                # Branch on clip method
                if self.weight_clip_threshold == 0.0:
                    clip_max_abs = find_clip_mmse(values, self.num_bits_params)
                    clip_max_abs_c = find_clip_mmse(values_c, self.num_bits_params)
                    if wrapped_module.projection_size is not 0:
                        clip_max_abs_proj = find_clip_mmse(values_proj, self.num_bits_proj)
                    if wrapped_module.bias is not None:
                        clip_bias_max_abs = find_clip_mmse(values_b, self.num_bits_params)

                elif self.weight_clip_threshold == -1.0:
                    clip_max_abs = find_clip_aciq(values, self.num_bits_params)
                    clip_max_abs_c = find_clip_aciq(values_c, self.num_bits_params)
                    if wrapped_module.projection_size is not 0:
                        clip_max_abs_proj = find_clip_aciq(values_proj, self.num_bits_proj)
                    if wrapped_module.bias is not None:
                        clip_bias_max_abs = find_clip_aciq(values_b, self.num_bits_params)
                elif self.weight_clip_threshold == -2.0:
                    clip_max_abs = find_clip_entropy(values, self.num_bits_params)
                    clip_max_abs_c = find_clip_entropy(values_c, self.num_bits_params)
                    if wrapped_module.projection_size is not 0:
                        clip_max_abs_proj = find_clip_entropy(values_proj, self.num_bits_proj)
                    if wrapped_module.bias is not None:
                        clip_bias_max_abs = find_clip_entropy(values_b, self.num_bits_params)
                else:
                    raise ValueError('Undefined weight clip method')

            self.w_scale = symmetric_linear_quantization_scale_factor(self.num_bits_params, clip_max_abs)
            self.w_scale_c = symmetric_linear_quantization_scale_factor(self.num_bits_params, clip_max_abs_c)
            if wrapped_module.projection_size is not 0:
                self.w_scale_proj = symmetric_linear_quantization_scale_factor(self.num_bits_proj, clip_max_abs_proj)
            if wrapped_module.bias is not None:
                self.b_scale = symmetric_linear_quantization_scale_factor(self.num_bits_params, clip_bias_max_abs)

            # Grid aware OCS
            q_weight_np, in_channels_to_split,in_channels_to_split_str = ocs_wts(
                    weight_np,
                    self.weight_expand_ratio,
                    split_threshold=self.split_threshold,
                    w_scale=self.w_scale,
                    grid_aware=True,Convolution=False)
            q_weight_c_np, in_channels_to_split_c,in_channels_to_split_str_c = ocs_wts(
                    weight_c_np,
                    self.weight_expand_ratio,
                    axis=0,
                    split_threshold=self.split_threshold,
                    w_scale=self.w_scale_c,
                    grid_aware=True,Convolution=False)
            if wrapped_module.projection_size is not 0:
                q_weight_proj_np, in_channels_to_split_proj,in_channels_to_split_str_proj = ocs_wts(
                    weight_proj_np,
                    self.weight_expand_ratio,
                    axis=0,
                    split_threshold=self.split_threshold,
                    w_scale=self.w_scale_c,
                    grid_aware=True,Convolution=False)

            # Save which channels got split
            if len(in_channels_to_split) > 0:
                assert(q_weight_np.shape[1] == num_channels + len(in_channels_to_split))
                self.channels_to_split = np.array(in_channels_to_split)
                self.channels_to_split = torch.from_numpy(self.channels_to_split).cuda()

                self.channels_to_split_c = np.array(in_channels_to_split_c)
                self.channels_to_split_c = torch.from_numpy(self.channels_to_split_c).cuda()

                if wrapped_module.projection_size is not 0:
                    self.channels_to_split_proj = np.array(in_channels_to_split_proj)
                    self.channels_to_split_proj = torch.from_numpy(self.channels_to_split_proj).cuda()

            q_weight_torch = torch.from_numpy(q_weight_np)
            q_weight_torch_org=q_weight_torch.clone()

            q_weight_c_torch = torch.from_numpy(q_weight_c_np)
            q_weight_c_torch_org=q_weight_c_torch.clone()

            if wrapped_module.projection_size is not 0:
                q_weight_proj_torch = torch.from_numpy(q_weight_proj_np)
                q_weight_proj_torch_org=q_weight_proj_torch.clone()

            if wrapped_module.bias is not None:
                q_bias_torch = wrapped_module.bias.data
            linear_quantize_clamp(q_weight_torch,
                                  self.w_scale,
                                  self.params_min_q_val,
                                  self.params_max_q_val,binary_flag,
                                  inplace=True)

            linear_quantize_clamp(q_weight_c_torch,
                                  self.w_scale_c,
                                  self.params_min_q_val,
                                  self.params_max_q_val,binary_flag,
                                  inplace=True)
            if wrapped_module.projection_size is not 0:
                linear_quantize_clamp(q_weight_proj_torch,
                                  self.w_scale_proj,
                                  self.params_min_q_val_proj,
                                  self.params_max_q_val_proj,binary_flag,
                                  inplace=True)

            if wrapped_module.bias is not None:
                linear_quantize_clamp(q_bias_torch,
                                  self.b_scale,
                                  self.params_min_q_val,
                                  self.params_max_q_val,binary_flag,
                                  inplace=True)
        #    print(compute_MSE(values,q_weight_np))
            wrapped_module.weight.data = q_weight_torch
            wrapped_module.split=in_channels_to_split_str

            wrapped_module.weight_c.data = q_weight_c_torch
            wrapped_module.split_c=in_channels_to_split_str_c

            if wrapped_module.projection_size is not 0:
                wrapped_module.weight_proj.data = q_weight_proj_torch
                wrapped_module.split_proj=in_channels_to_split_str_proj



    def forward(self, *inputs):
        if PROFILE_MODE == True:
            """  profiling """
            # Flag indicating whether profiling was done
            self.profile_info = True
            # Number of inputs should be 1
            assert(len(inputs) == 1)
            input = inputs[0]
            # Input shape is (Batch, C, H, W)
            self.input_shape = input.shape
            num_channels = self.input_shape[1]
            #print(input.shape)

            if num_channels > 3:

                torch_input_tensor = input.data.cpu()
                input_np = torch_input_tensor.numpy()

                """ Clipping """
                if self.act_clip_threshold > 0.0:
                    act_clip_max_abs = np.max(np.abs(input_np)) * self.act_clip_threshold
                else:
                #    print('Auto-tuning for activation clip threshold...')
                    if self.act_clip_threshold == 0.0:
                        act_clip_max_abs = find_clip_mmse(input_np.flatten(), self.num_bits_acts)
                    elif self.act_clip_threshold == -1.0:
                        act_clip_max_abs = find_clip_aciq(input_np.flatten(), self.num_bits_acts)
                    elif self.act_clip_threshold == -2.0:
                        act_clip_max_abs = find_clip_entropy(input_np.flatten(), self.num_bits_acts)
                    else:
                        raise ValueError('Undefined act clip method')

                self.act_clip_max_abs = torch.tensor(act_clip_max_abs).cuda()

                """ Get channels to split """
                # Unused

            # For profiling, we use the original FP weights
            weight_q = self.wrapped_module.weight.data
            self.wrapped_module.weight.data = self.weight_orig.cuda()
            # Run the forward pass
            accum = self.wrapped_module.forward(*inputs)
            self.wrapped_module.weight.data = weight_q
            return accum
        else:
            assert(self.profile_info)
            assert(len(inputs) == 1)

            """ Avoid quantizing the input layer """
            input = inputs[0]
            num_channels = input.shape[1]
            if num_channels == 3:
                return self.wrapped_module.forward(*inputs)

            """ If we don't need OCS on the activations, skip the
                cuda->cpu->cuda transfer to save time """
            #if self.act_expand_ratio == 0.0:
            #    return super(OCSParamLayerWrapper, self).forward(input)

            """ Quantize inputs """
            inputs_q = []
            for idx, input in enumerate(inputs):
                # Clip
                new_max_abs = self.act_clip_max_abs

                # Determine scale factor for quantization
                in_scale = symmetric_linear_quantization_scale_factor(
                            self.num_bits_acts, new_max_abs)
                self.current_accum_scale = in_scale * self.w_scale

                input_q = linear_quantize_clamp(input.data, in_scale,
                                      self.acts_min_q_val, self.acts_max_q_val,
                                      inplace=False)

                # Duplicate channels
                if self.channels_to_split is not None:
                    N, _, H, W = input.shape

                    input_splits = torch.index_select(input_q, dim=1, index=self.channels_to_split)

                    input_ocs = torch.cat([input_q, input_splits], dim=1)
                else:
                    input_ocs = input_q

                inputs_q.append(torch.autograd.Variable(input_ocs))

            """ forward through wrapped module """
            accum = self.wrapped_module.forward(*inputs_q)
            clamp(accum.data, self.accum_min_q_val, self.accum_max_q_val, inplace=True)
            """ re-quantize accumulator to quantized output range """
            requant_scale, out_scale = self.post_quantized_forward(accum)
            out_q = linear_quantize_clamp(accum.data, requant_scale, self.acts_min_q_val, self.acts_max_q_val, inplace=True)
            """ de-quantize back to FP32 """
            out_f = linear_dequantize(out_q, out_scale, inplace=True)
            return torch.autograd.Variable(out_f)

    def post_quantized_forward(self, accumulator):
        accum_max_abs = get_tensor_max_abs(accumulator)
        y_f_max_abs = accum_max_abs / self.current_accum_scale
        out_scale = symmetric_linear_quantization_scale_factor(self.num_bits_acts, y_f_max_abs)
        requant_scale = out_scale / self.current_accum_scale
        return requant_scale, out_scale

    def pre_quantized_forward(self, input):
        in_scale = symmetric_linear_quantization_scale_factor(self.num_bits_acts, get_tensor_max_abs(input))
        self.current_accum_scale = in_scale * self.w_scale
        return [in_scale]


    def __repr__(self):
        tmpstr = self.__class__.__name__ + '(\n'
        tmpstr += '  (wrapped_module): ' + self.wrapped_module.__repr__() + '\n'
        tmpstr += '  num_bits_activations={0}, num_bits_parameters={1}'.format(
            self.num_bits_acts, self.num_bits_params) + '\n'
        tmpstr += ')'
        return tmpstr


class OCSQuantizer(Quantizer):
    def __init__(self, model, bits_activations=8, bits_parameters=8,
                 weight_expand_ratio=0.0, weight_clip_threshold=1.0,
                 act_expand_ratio=0.0, act_clip_threshold=1.0,skip_layer_one=False, layer_one_size=69):
        super(OCSQuantizer, self).__init__(model, bits_activations=bits_activations,
                                           bits_weights=bits_parameters,
                                           train_with_fp_copy=False)
        self.model.quantizer_metadata = {'type': type(self),
                                         'params': {'bits_activations': bits_activations,
                                                    'bits_parameters': bits_parameters}}

        def replace_fn(module, name, qbits_map):
            return OCSParamLayerWrapper(module, qbits_map[name].acts, qbits_map[name].wts,
                                        weight_expand_ratio=weight_expand_ratio, weight_clip_threshold=weight_clip_threshold,
                                        act_expand_ratio=act_expand_ratio, act_clip_threshold=act_clip_threshold,skip_layer_one=skip_layer_one, layer_one_size=layer_one_size)

        self.replacement_factory[nn.Conv2d] = replace_fn
        self.replacement_factory[nn.Linear] = replace_fn
        self.replacement_factory[sru.SRUCell] = replace_fn
