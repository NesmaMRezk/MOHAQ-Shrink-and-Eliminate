#Nesma Rezk July 2020
#The functions in this file translates the pretrained model formats between Pytorch Kaldi and NWDistiller formats and quantizes the model.
import sys
from distutils.util import strtobool

import torch

import quantization
from Tensorboard import Tensorboard
from quantization.clipped_linear import *
import numpy as np

#Nesma M. Rezk
#checkpoints generated from pytorch kaldi are used for quantiztion. Input files are named checkpoint_architectureX.pth.tar
#After quantization, model is wrapped in a format that is accepted by pytrchkaldi library
#The quantized parameters replaces the parameters in files named final_architectureX.pth.tar
#The quantized model are placed in the same folder and named quantized_TIMIT_architectureX.pth.tar
#quantize function takes inputs
#directory: the relative path of the input model.
#index: the index of the architecture file
#type: #type of model , SRU or LSTM or MLP
#number of bits: array of number of bits for each layer, if number of bits is 16, the layer is quantized to 16 fixed point. if less than 16 this will be linear integer qquantization
#number of projection bits: used for sru model
#this version has not been tested on LSTM

#folder='/usr/local/home/nesrez/pytorch-kaldi/exp/TIMIT_SRU_fbank'

def quantize_16_parameter(vector):

    to_quantize = np.array(vector.data)
    if get_tensor_max_abs(vector) < 2:
        # maximum is 1.xxxx
        # 1 bit for sign, 1 bit for whole, 14 bit for fraction part
        quantized_vector = np.round(to_quantize * 10000) / 10000
    elif get_tensor_max_abs(vector) < 32:
        # maximum is 31.xxxx
        # 1 bit for sign, 5 bit for whole, 10 bit for fraction part
        quantized_vector = np.round(to_quantize * 1000) / 1000
    else:
        # 1 bit for sign, 8 bit for whole, 7 bit for fraction part
        quantized_vector = np.round(to_quantize * 100) / 100

    quantized = torch.from_numpy(quantized_vector)
    return quantized


def wrap_MLP(model, before, n_bits, ):

    layer_flags=[True,True,True,True]
    for i in range(model.N_dnn_lay):
        if n_bits[i]==16:
            layer_flags[i]=False

        #if layer_flags[i]==True:
      #  file1.writelines(str(model.wx[i].w_scale)+"\n")
        model.wx[i] = model.wx[i].wrapped_module
       # file1.writelines(model.wx[i].split+"\n")

        if layer_flags[i]==False:
            before.cpu()
            model.wx[i].weight.data= quantize_16_parameter(before.wx[i].weight)
            model.wx[i].bias.data = quantize_16_parameter(before.wx[i].bias)
            model.cuda()
            before.cuda()

       # file1.write(" \n")
    return

#not tested/working after changes in quantize function for SRU and MLP
def wrap_LSTM(model, before, n_bits,skip=0):
    xlayer_flags=[True,True,True,True]
    hlayer_flags=[True,True,True,True]

    for i in range(model.N_lstm_lay):
        if n_bits[i] == 16:
            xlayer_flags[i] = False
        if n_bits[i+model.N_lstm_lay] == 16:
            hlayer_flags[i] = False

        before.cpu()


        model.wfx[i] = model.wfx[i].wrapped_module

        model.wix[i] = model.wix[i].wrapped_module

        model.wox[i] = model.wox[i].wrapped_module

        model.wcx[i] = model.wcx[i].wrapped_module

        model.ufh[i] = model.ufh[i].wrapped_module

        model.uih[i] = model.uih[i].wrapped_module

        model.uoh[i] = model.uoh[i].wrapped_module

        model.uch[i] = model.uch[i].wrapped_module



        if xlayer_flags[i]==False:
            model.wfx[i].weight.data =quantize_16_parameter( before.wfx[i].weight)
            model.wix[i].weight.data =quantize_16_parameter( before.wix[i].weight)
            model.wcx[i].weight.data =quantize_16_parameter( before.wcx[i].weight)
            model.wox[i].weight.data = quantize_16_parameter(before.wox[i].weight)
            if model.wfx[i].bias!= None:
                model.wfx[i].bias.data = quantize_16_parameter(before.wfx[i].bias)
                model.wix[i].bias.data = quantize_16_parameter(before.wix[i].bias)
                model.wcx[i].bias.data = quantize_16_parameter(before.wcx[i].bias)
                model.wox[i].bias.data = quantize_16_parameter(before.wox[i].bias)
        if hlayer_flags[i]==False:
            model.ufh[i].weight.data =quantize_16_parameter( before.ufh[i].weight)
            model.uih[i].weight.data =quantize_16_parameter( before.uih[i].weight)
            model.uch[i].weight.data =quantize_16_parameter( before.uch[i].weight)
            model.uoh[i].weight.data =quantize_16_parameter( before.uoh[i].weight)
        if skip==1:
           model.uch[i].weight.data = quantize_16_parameter(before.uch[i].weight)
        elif skip==2:
           if i==0 or i==1:
               model.uch[i].weight.data = quantize_16_parameter(before.uch[i].weight)


        model.bn_wfx[i].weight.data= quantize_16_parameter( before.bn_wfx[i].weight)
        model.bn_wfx[i].running_mean.data = quantize_16_parameter(before.bn_wfx[i].running_mean)
        model.bn_wfx[i].running_var.data = quantize_16_parameter(before.bn_wfx[i].running_var)
        model.bn_wfx[i].bias.data = quantize_16_parameter(before.bn_wfx[i].bias)

        model.bn_wcx[i].weight.data= quantize_16_parameter( before.bn_wcx[i].weight)
        model.bn_wcx[i].running_mean.data = quantize_16_parameter(before.bn_wcx[i].running_mean)
        model.bn_wcx[i].running_var.data = quantize_16_parameter(before.bn_wcx[i].running_var)
        model.bn_wcx[i].bias.data = quantize_16_parameter(before.bn_wcx[i].bias)

        model.bn_wix[i].weight.data= quantize_16_parameter( before.bn_wix[i].weight)
        model.bn_wix[i].running_mean.data = quantize_16_parameter(before.bn_wix[i].running_mean)
        model.bn_wix[i].running_var.data = quantize_16_parameter(before.bn_wix[i].running_var)
        model.bn_wix[i].bias.data = quantize_16_parameter(before.bn_wix[i].bias)

        model.bn_wox[i].weight.data= quantize_16_parameter( before.bn_wox[i].weight)
        model.bn_wox[i].running_mean.data = quantize_16_parameter(before.bn_wox[i].running_mean)
        model.bn_wox[i].running_var.data = quantize_16_parameter(before.bn_wox[i].running_var)
        model.bn_wox[i].bias.data = quantize_16_parameter(before.bn_wox[i].bias)

        model.cuda()
        before.cuda()
    return

def wrap_GRU(model, before, n_bits,skip=0):
    xlayer_flags=[True,True,True,True,True]
    hlayer_flags=[True,True,True,True,True]

    for i in range(model.N_gru_lay):
        if n_bits[i] == 16:
            xlayer_flags[i] = False
        if n_bits[i+model.N_gru_lay] == 16:
            hlayer_flags[i] = False

        before.cpu()


        model.wh[i] = model.wh[i].wrapped_module

        model.wz[i] = model.wz[i].wrapped_module

        model.wr[i] = model.wr[i].wrapped_module



        model.uh[i] = model.uh[i].wrapped_module

        model.ur[i] = model.ur[i].wrapped_module

        model.uz[i] = model.uz[i].wrapped_module


        if xlayer_flags[i]==False:
            model.wh[i].weight.data =quantize_16_parameter( before.wh[i].weight)
            model.wr[i].weight.data =quantize_16_parameter( before.wr[i].weight)
            model.wz[i].weight.data =quantize_16_parameter( before.wz[i].weight)

            if model.wh[i].bias!= None:
                model.wh[i].bias.data = quantize_16_parameter(before.wh[i].bias)
                model.wr[i].bias.data = quantize_16_parameter(before.wr[i].bias)
                model.wz[i].bias.data = quantize_16_parameter(before.wz[i].bias)

        if hlayer_flags[i]==False:
            model.uh[i].weight.data =quantize_16_parameter( before.uh[i].weight)
            model.ur[i].weight.data =quantize_16_parameter( before.ur[i].weight)
            model.uz[i].weight.data =quantize_16_parameter( before.uz[i].weight)

        if skip==1:
            model.uh[i].weight.data = quantize_16_parameter(before.uh[i].weight)
        if skip==2:
            if i==0 or i==1 or i==2:
                model.uh[i].weight.data = quantize_16_parameter(before.uh[i].weight)

        model.bn_wh[i].weight.data= quantize_16_parameter( before.bn_wh[i].weight)
        model.bn_wh[i].running_mean.data = quantize_16_parameter(before.bn_wh[i].running_mean)
        model.bn_wh[i].running_var.data = quantize_16_parameter(before.bn_wh[i].running_var)
        model.bn_wh[i].bias.data = quantize_16_parameter(before.bn_wh[i].bias)

        model.bn_wr[i].weight.data= quantize_16_parameter( before.bn_wr[i].weight)
        model.bn_wr[i].running_mean.data = quantize_16_parameter(before.bn_wr[i].running_mean)
        model.bn_wr[i].running_var.data = quantize_16_parameter(before.bn_wr[i].running_var)
        model.bn_wr[i].bias.data = quantize_16_parameter(before.bn_wr[i].bias)

        model.bn_wz[i].weight.data= quantize_16_parameter( before.bn_wz[i].weight)
        model.bn_wz[i].running_mean.data = quantize_16_parameter(before.bn_wz[i].running_mean)
        model.bn_wz[i].running_var.data = quantize_16_parameter(before.bn_wz[i].running_var)
        model.bn_wz[i].bias.data = quantize_16_parameter(before.bn_wz[i].bias)

        model.cuda()
        before.cuda()
    return
def wrap_liGRU(model, before, n_bits,skip=0):
    xlayer_flags=[True,True,True,True,True]
    hlayer_flags=[True,True,True,True,True]

    for i in range(model.N_ligru_lay):
        if n_bits[i] == 16:
            xlayer_flags[i] = False
        if n_bits[i+model.N_ligru_lay] == 16:
            hlayer_flags[i] = False

        before.cpu()


        model.wh[i] = model.wh[i].wrapped_module

        model.wz[i] = model.wz[i].wrapped_module


        model.uz[i] = model.uz[i].wrapped_module

        model.uh[i] = model.uh[i].wrapped_module
        if xlayer_flags[i]==False:

            model.wh[i].weight.data =quantize_16_parameter( before.wh[i].weight)
            model.wz[i].weight.data =quantize_16_parameter( before.wz[i].weight)

            if model.wh[i].bias!= None:
                model.wh[i].bias.data = quantize_16_parameter(before.wh[i].bias)
                model.wz[i].bias.data = quantize_16_parameter(before.wz[i].bias)

        if hlayer_flags[i]==False:
            model.uh[i].weight.data =quantize_16_parameter( before.uh[i].weight)
            model.uz[i].weight.data =quantize_16_parameter( before.uz[i].weight)
        if skip==1:
            model.uh[i].weight.data = quantize_16_parameter(before.uh[i].weight)
        elif skip==2:
            if i==0 or i==1 or i==2:
                model.uh[i].weight.data = quantize_16_parameter(before.uh[i].weight)


        model.bn_wh[i].weight.data= quantize_16_parameter( before.bn_wh[i].weight)
        model.bn_wh[i].running_mean.data = quantize_16_parameter(before.bn_wh[i].running_mean)
        model.bn_wh[i].running_var.data = quantize_16_parameter(before.bn_wh[i].running_var)
        model.bn_wh[i].bias.data = quantize_16_parameter(before.bn_wh[i].bias)

        model.bn_wz[i].weight.data= quantize_16_parameter( before.bn_wz[i].weight)
        model.bn_wz[i].running_mean.data = quantize_16_parameter(before.bn_wz[i].running_mean)
        model.bn_wz[i].running_var.data = quantize_16_parameter(before.bn_wz[i].running_var)
        model.bn_wz[i].bias.data = quantize_16_parameter(before.bn_wz[i].bias)

        model.cuda()
        before.cuda()
    return
def wrap_SRU(model, before, n_bits,p_bits, ):


    layer_flags=[True,True,True,True,True,True,True,True,True,True,True,True]
    proj_flags=[True,True,True,True,True,True,True,True,True,True,True,True]
    for i in range(model.num_layers):
        if n_bits[i]==16:
            layer_flags[i]=False
        if p_bits[i] == 16:
            proj_flags[i] = False

        before.cpu()
        model.sru.rnn_lst[i] = model.sru.rnn_lst[i].wrapped_module
        model.sru.rnn_lst[i].bias.data = quantize_16_parameter(before.sru.rnn_lst[i].bias)
        model.sru.rnn_lst[i].weight_c.data = quantize_16_parameter(before.sru.rnn_lst[i].weight_c)
        if proj_flags[i] == False and before.sru.rnn_lst[i].weight_proj != None:
            model.sru.rnn_lst[i].weight_proj.data = quantize_16_parameter(before.sru.rnn_lst[i].weight_proj)

        if layer_flags[i] == False:
            model.sru.rnn_lst[i].weight.data = quantize_16_parameter(before.sru.rnn_lst[i].weight)

    model.cuda()
    before.cuda()
    return

def quantize_inline(model, before,type, n_bits,p_bits=[],checkpoint=None,n_layers=4,skip=0):
    bits=[]
    if type=='SRU':
        for i in range(n_layers):
            bits.append(n_bits[i])
            bits.append(p_bits[i])
       # bits=[n_bits[0],p_bits[0],n_bits[1],p_bits[1],n_bits[2],p_bits[2],n_bits[3],p_bits[3]]
    elif type=='LSTM':
        for i in range(4):
            for j in range(n_layers*2):
                bits.append(n_bits[j])

    elif type == 'GRU':
        for i in range(3):
            for j in range(n_layers * 2):
                bits.append(n_bits[j])

    elif type == 'liGRU':
        for i in range(2):
            for j in range(n_layers * 2):
                bits.append(n_bits[j])

        #bits=[n_bits[0],n_bits[0],n_bits[0],n_bits[0],n_bits[1],n_bits[1],n_bits[1],n_bits[1],n_bits[2],n_bits[2],n_bits[2],n_bits[2],n_bits[3],n_bits[3],n_bits[3],n_bits[3]]

    else:
        bits.append(n_bits[0])
    quantizer = quantization.OCSQuantizer(model,
                  bits_activations=8, bits_parameters=bits, # activations bits, weights bits
                  weight_expand_ratio=0, # weight expand ratio 0,0.01,0.02,0.05
                  weight_clip_threshold=0,# weight clip threshold >1: used as it is, 0: MMSE, -1: ACIQ, -2:cross entropy
                  act_expand_ratio=0, # activation expand ratio-> not used
                  act_clip_threshold=8,  #activation clip threshold ->not used
                  skip_layer_one=False, #-> not used
                  layer_one_size=69)   #->not used
    model.cpu()
    quantizer.prepare_model()
    model.cuda()

   # file1 = open(directory+"/OCS"+str(index)+".txt", "w")

    if type=='MLP':
        wrap_MLP(model, before,n_bits )
    elif type=='LSTM':

        wrap_LSTM(model, before,n_bits,skip)
    elif type=='GRU':

        wrap_GRU(model, before,n_bits,skip)
    elif type=='liGRU':

        wrap_liGRU(model, before,n_bits,skip)
    elif type=='SRU':
        wrap_SRU(model, before,n_bits,p_bits)

    #file1.close()  # to change file access modes
    if checkpoint!=None:
        checkpoint["model_par"] = model.state_dict()
    return model

def quantize(directory, index, type, n_bits,p_bits=[],n_layers=4,skip=0):
    kaldi_model = directory + '/exp_files/checkpoint_architecture' + str(index) + '.pth.tar'
    kaldi_model_t = directory + '/exp_files/final_architecture' + str(index) + '.pth.tar'
    model = torch.load(kaldi_model)  # model to quantize
    model.load_state_dict(torch.load(kaldi_model_t)["model_par"])
    before = torch.load(kaldi_model)  # model not quantized to use in wrapping
    before.load_state_dict(torch.load(kaldi_model_t)["model_par"])

    originalmodel=torch.load(directory+'/exp_files/final_architecture'+str(index)+'.pth.tar')
    originalmodel["model_par"]=quantize_inline(model,before, type, n_bits, p_bits,n_layers=n_layers,skip=skip).state_dict()
    torch.save(originalmodel,directory+'/exp_files/quantize_TIMIT_architecture'+str(index)+'.pth.tar')
    print("hi")
    return
