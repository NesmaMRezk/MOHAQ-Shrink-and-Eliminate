##########################################################
# pytorch-kaldi v.0.1
# Mirco Ravanelli, Titouan Parcollet
# Mila, University of Montreal
# October 2018
##########################################################
#Nesma: Similar to run_exp but for Testing only, training and validation are removed

from __future__ import print_function

import os

import sys
import glob
import torch
import configparser
import numpy as np

from quantize import quantize
from utils import (
    check_cfg,
    create_lists,
    create_configs,
    compute_avg_performance,
    read_args_command_line,
    run_shell,
    compute_n_chunks,
    get_all_archs,
    cfg_item2sec,
    dump_epoch_results,
    create_curves,
    change_lr_cfg,
    expand_str_ep,
    do_validation_after_chunk,
    get_val_info_file_path,
    get_val_cfg_file_path,
    get_chunks_after_which_to_validate,
)
from data_io import read_lab_fea_refac01 as read_lab_fea
from shutil import copyfile
from core import read_next_chunk_into_shared_list_with_subprocess, extract_data_from_shared_list, convert_numpy_to_torch
import re
from distutils.util import strtobool
import importlib
import math
import multiprocessing


def _run_forwarding_in_subprocesses(config):
    use_cuda = strtobool(config["exp"]["use_cuda"])
    if use_cuda:
        return False
    else:
        return True


def _is_first_validation(ep, ck, N_ck_tr, config):
    def _get_nr_of_valid_per_epoch_from_config(config):
        if not "nr_of_valid_per_epoch" in config["exp"]:
            return 1
        return int(config["exp"]["nr_of_valid_per_epoch"])
    
    if ep>0:
        return False
    
    val_chunks = get_chunks_after_which_to_validate(N_ck_tr, _get_nr_of_valid_per_epoch_from_config(config))
    if ck == val_chunks[0]:
        return True

    
    return False


def _max_nr_of_parallel_forwarding_processes(config):
    if "max_nr_of_parallel_forwarding_processes" in config["forward"]:
        return int(config["forward"]["max_nr_of_parallel_forwarding_processes"])
    return -1

#Nesma put argument here
# Reading global cfg file (first argument-mandatory file)
#Nesma: this function is almost the same as in pytorchkaldi
# I only made it only for inference and removed the training part
#I also added a return value which is the WER taken from the displayed output
def run_inference(cfg_file):
    if not (os.path.exists(cfg_file)):
        sys.stderr.write("ERROR: The config file %s does not exist!\n" % (cfg_file))
        sys.exit(0)
    else:
        config = configparser.ConfigParser()
        config.read(cfg_file)


    # Reading and parsing optional arguments from command line (e.g.,--optimization,lr=0.002)
    [section_args, field_args, value_args] = read_args_command_line(sys.argv, config)


    # Output folder creation
    out_folder = config["exp"]["out_folder"]
    if not os.path.exists(out_folder):
        os.makedirs(out_folder + "/exp_files")

    # Log file path
    log_file = config["exp"]["out_folder"] + "/log.log"


    # Read, parse, and check the config file
    cfg_file_proto = config["cfg_proto"]["cfg_proto"]
    [config, name_data, name_arch] = check_cfg(cfg_file, config, cfg_file_proto)


    # Read cfg file options
    is_production = strtobool(config["exp"]["production"])

    cfg_file_proto_chunk = config["cfg_proto"]["cfg_proto_chunk"]

    cmd = config["exp"]["cmd"]
    N_ep = int(config["exp"]["N_epochs_tr"])
    N_ep_str_format = "0" + str(max(math.ceil(np.log10(N_ep)), 1)) + "d"
    tr_data_lst = config["data_use"]["train_with"].split(",")
    valid_data_lst = config["data_use"]["valid_with"].split(",")
    forward_data_lst = config["data_use"]["forward_with"].split(",")
    max_seq_length_train = config["batches"]["max_seq_length_train"]
    forward_save_files = list(map(strtobool, config["forward"]["save_out_file"].split(",")))


    print("- Reading config file......OK!")

    # Copy the global cfg file into the output folder
    cfg_file = cfg_file
    with open(cfg_file, "w") as configfile:
        config.write(configfile)


    # Load the run_nn function from core libriary
    # The run_nn is a function that process a single chunk of data
    run_nn_script = config["exp"]["run_nn_script"].split(".py")[0]
    module = importlib.import_module("core")
    run_nn = getattr(module, run_nn_script)


    # Splitting data into chunks (see out_folder/additional_files)
    create_lists(config)

    # Writing the config files
    create_configs(config)

    print("- Chunk creation......OK!\n")

    # create res_file
    res_file_path = out_folder + "/res.res"
    res_file = open(res_file_path, "w")
    res_file.close()


    # Learning rates and architecture-specific optimization parameters
    arch_lst = get_all_archs(config)
    lr = {}
    auto_lr_annealing = {}
    improvement_threshold = {}
    halving_factor = {}
    pt_files = {}

    for arch in arch_lst:
        lr[arch] = expand_str_ep(config[arch]["arch_lr"], "float", N_ep, "|", "*")
        if len(config[arch]["arch_lr"].split("|")) > 1:
            auto_lr_annealing[arch] = False
        else:
            auto_lr_annealing[arch] = True
        improvement_threshold[arch] = float(config[arch]["arch_improvement_threshold"])
        halving_factor[arch] = float(config[arch]["arch_halving_factor"])
        pt_files[arch] = config[arch]["arch_pretrain_file"]


    # If production, skip training and forward directly from last saved models
    if is_production:
        ep = N_ep
        N_ep = 0
        model_files = {}

        for arch in pt_files.keys():
            model_files[arch] = out_folder + "/exp_files/final_" + arch + ".pth.tar"


    #op_counter = 23  # used to dected the next configuration file from the list_chunks.txt

    # Reading the ordered list of config file to process
    cfg_file_list = [line.rstrip("\n") for line in open(out_folder + "/exp_files/list_chunks.txt")]
    cfg_file_list.append(cfg_file_list[-1])


    # A variable that tells if the current chunk is the first one that is being processed:
    processed_first = True

    data_name = []
    data_set = []
    data_end_index = []
    fea_dict = []
    lab_dict = []
    arch_dict = []


    # --------FORWARD--------#

    ep=N_ep-1
    for forward_data in forward_data_lst:

        # Compute the number of chunks
        N_ck_forward = 1#compute_n_chunks(out_folder, forward_data, ep, N_ep_str_format, "forward")
        print(N_ck_forward)
        N_ck_str_format = "0" + str(max(math.ceil(np.log10(N_ck_forward)), 1)) + "d"

        processes = list()
        info_files = list()

        for ck in range(N_ck_forward):

            if not is_production:
                print("Testing %s chunk = %i / %i" % (forward_data, ck + 1, N_ck_forward))
            else:
                print("Forwarding %s chunk = %i / %i" % (forward_data, ck + 1, N_ck_forward))

            # output file
            info_file = (
                out_folder
                + "/exp_files/forward_"
                + forward_data
                + "_ep"
                + format(ep, N_ep_str_format)
                + "_ck"
                + format(ck, N_ck_str_format)
                + ".info"
            )
            config_chunk_file=out_folder+"/quantize_TIMIT.cfg"
            next_config_file = out_folder + "/quantize_TIMIT.cfg"
         #   config_chunk_file = (
         #       out_folder
         #       + "/exp_files/forward_"
         #       + forward_data
         #       + "_ep"
         #       + format(ep, N_ep_str_format)
         #       + "_ck"
         #       + format(ck, N_ck_str_format)
         #       + ".cfg"
         #   )

            # Do forward if the chunk was not already processed
        #    if not (os.path.exists(info_file)):

            # Doing forward

            # getting the next chunk
           # next_config_file = out_folder+"exp_files/forward_TIMIT_test_ep"+str(N_ep-1)+"_ck0.cfg"

            # run chunk processing
            if _run_forwarding_in_subprocesses(config):
                shared_list = list()
                output_folder = config["exp"]["out_folder"]
                save_gpumem = strtobool(config["exp"]["save_gpumem"])
                use_cuda = strtobool(config["exp"]["use_cuda"])
                p = read_next_chunk_into_shared_list_with_subprocess(
                    read_lab_fea, shared_list, config_chunk_file, is_production, output_folder, wait_for_process=True
                )
                data_name, data_end_index_fea, data_end_index_lab, fea_dict, lab_dict, arch_dict, data_set_dict = extract_data_from_shared_list(
                    shared_list
                )

                data_set_inp, data_set_ref = convert_numpy_to_torch(data_set_dict, save_gpumem, use_cuda)
                data_set = {"input": data_set_inp, "ref": data_set_ref}
                data_end_index = {"fea": data_end_index_fea, "lab": data_end_index_lab}
                p = multiprocessing.Process(
                    target=run_nn,
                    kwargs={
                        "data_name": data_name,
                        "data_set": data_set,
                        "data_end_index": data_end_index,
                        "fea_dict": fea_dict,
                        "lab_dict": lab_dict,
                        "arch_dict": arch_dict,
                        "cfg_file": config_chunk_file,
                        "processed_first": False,
                        "next_config_file": None,
                    },
                )
                processes.append(p)
                if _max_nr_of_parallel_forwarding_processes(config) != -1 and len(
                    processes
                ) > _max_nr_of_parallel_forwarding_processes(config):
                    processes[0].join()
                    del processes[0]
                p.start()
            else:
                [data_name, data_set, data_end_index, fea_dict, lab_dict, arch_dict] = run_nn(
                    data_name,
                    data_set,
                    data_end_index,
                    fea_dict,
                    lab_dict,
                    arch_dict,
                    config_chunk_file,
                    processed_first,
                    next_config_file
                )
                processed_first = False
                #Nesma: No need for info file during testing, commented the code
                if not (os.path.exists(info_file)):
                    sys.stderr.write(
                        "ERROR: forward chunk %i of dataset %s not done! File %s does not exist.\nSee %s \n"
                        % (ck, forward_data, info_file, log_file)
                    )
                    sys.exit(0)
            info_files.append(info_file)

        if _run_forwarding_in_subprocesses(config):
            for process in processes:
                process.join()
            for info_file in info_files:
                if not (os.path.exists(info_file)):
                    sys.stderr.write(
                        "ERROR: File %s does not exist. Forwarding did not suceed.\nSee %s \n" % (info_file, log_file)
                    )
                    sys.exit(0)

    # --------DECODING--------#
    dec_lst = glob.glob(out_folder + "/exp_files/*_to_decode.ark")
    print('dec_lst')
    print(dec_lst)
    forward_data_lst = config["data_use"]["forward_with"].split(",")
    forward_outs = config["forward"]["forward_out"].split(",")
    forward_dec_outs = list(map(strtobool, config["forward"]["require_decoding"].split(",")))


    for data in forward_data_lst:

        for k in range(len(forward_outs)):
            if forward_dec_outs[k]:

                print("Decoding %s output %s" % (data, forward_outs[k]))

                info_file = out_folder + "/exp_files/decoding_" + data + "_" + forward_outs[k] + ".info"

                # create decode config file
                config_dec_file = out_folder + "/decoding_" + data + "_" + forward_outs[k] + ".conf"
                config_dec = configparser.ConfigParser()
                config_dec.add_section("decoding")

                for dec_key in config["decoding"].keys():
                    config_dec.set("decoding", dec_key, config["decoding"][dec_key])

                # add graph_dir, datadir, alidir
                lab_field = config[cfg_item2sec(config, "data_name", data)]["lab"]

                # Production case, we don't have labels
                if not is_production:
                    pattern = "lab_folder=(.*)\nlab_opts=(.*)\nlab_count_file=(.*)\nlab_data_folder=(.*)\nlab_graph=(.*)"

                    alidir = re.findall(pattern, lab_field)[0][0]
                    config_dec.set("decoding", "alidir", os.path.abspath(alidir))

                    datadir = re.findall(pattern, lab_field)[0][3]
                    config_dec.set("decoding", "data", os.path.abspath(datadir))

                    graphdir = re.findall(pattern, lab_field)[0][4]
                    config_dec.set("decoding", "graphdir", os.path.abspath(graphdir))
                else:
                    pattern = "lab_data_folder=(.*)\nlab_graph=(.*)"
                    datadir = re.findall(pattern, lab_field)[0][0]
                    config_dec.set("decoding", "data", os.path.abspath(datadir))

                    graphdir = re.findall(pattern, lab_field)[0][1]
                    config_dec.set("decoding", "graphdir", os.path.abspath(graphdir))

                    # The ali dir is supposed to be in exp/model/ which is one level ahead of graphdir
                    alidir = graphdir.split("/")[0 : len(graphdir.split("/")) - 1]
                    alidir = "/".join(alidir)
                    config_dec.set("decoding", "alidir", os.path.abspath(alidir))

                with open(config_dec_file, "w") as configfile:
                    config_dec.write(configfile)

                out_folder = os.path.abspath(out_folder)
                files_dec = out_folder + "/exp_files/forward_" + data + "_ep*_ck*_" + forward_outs[k] + "_to_decode.ark"
                out_dec_folder = out_folder + "/decode_" + data + "_" + forward_outs[k]



                if not (os.path.exists(info_file)):

                    # Run the decoder
                    cmd_decode = (
                        cmd
                        + config["decoding"]["decoding_script_folder"]
                        + "/"
                        + config["decoding"]["decoding_script"]
                        + " "
                        + os.path.abspath(config_dec_file)
                        + " "
                        + out_dec_folder
                        + ' "'
                        + files_dec
                        + '"'
                    )

                    run_shell(cmd_decode, log_file)


                    # remove ark files if needed
                    if not forward_save_files[k]:
                        list_rem = glob.glob(files_dec)
                        for rem_ark in list_rem:
                            os.remove(rem_ark)

                # Print WER results and write info file

                cmd_res = "./check_res_dec.sh " + out_dec_folder
                wers = run_shell(cmd_res, log_file).decode("utf-8")
                res_file = open(res_file_path, "a")
                res_file.write("%s\n" % wers)
                print(wers)
    return wers[5:9]
#Nesma: this function set the activation bits in the configuration file. This value will be used during inference
def set_bits_conf(config, precisions, delta=[], n_layers=4):
    weight_bits_str = ""
    weight_act_str = ""
    proj_bits_str = ""
    proj_act_str = ""
    for i in range(n_layers):
        weight_bits_str = weight_bits_str + str(precisions[i])
        if i < n_layers - 1:
            weight_bits_str = weight_bits_str + ","
    for i in range(n_layers):
        weight_act_str = weight_act_str + str(precisions[i + n_layers])
        if i < n_layers - 1:
            weight_act_str = weight_act_str + ","

    for i in range(n_layers - 1):
        proj_bits_str = proj_bits_str + str(precisions[i + 2 * n_layers])
        if i < n_layers - 2:
            proj_bits_str = proj_bits_str + ","
    for i in range(n_layers - 1):
        proj_act_str = proj_act_str + str(precisions[i + 3 * n_layers - 1])
        if i < n_layers - 2:
            proj_act_str = proj_act_str + ","

    delta_str = ""
    j = 0
    for i in delta:
        j += 1
        if j < len(delta):
            delta_str = delta_str + str(i)

        if j < len(delta) - 1:
            delta_str = delta_str + ","

    config.set("architecture1", "delta", delta_str)

    config.set("architecture2", "delta", str(delta[len(delta) - 1]))
    config.set("architecture1", "weights_bits", weight_bits_str)
    config.set("architecture1", "inputs_bits", "16," + proj_act_str)
    config.set("architecture1", "projection_bits", "16," + proj_bits_str)
    config.set("architecture1", "weights_activation_bits", weight_act_str)
    config.set("architecture1", "output_bits", str(precisions[len(precisions) - 1]))
    print(str(precisions[n_layers * 4 - 1]))
    #  config.set("architecture1", "delta",str(delta[0]) + "," + str(delta[1]) + "," + str(delta[2]) + "," + str(delta[3])+","+str(delta[4]) + "," + str(delta[5]) + "," + str(delta[6]))
    config.set("architecture2", "number_of_bits", str(precisions[len(precisions) - 2]))
    config.set("architecture2", "weights", str(precisions[len(precisions) - 2]))


#   config.set("architecture2", "delta", str(delta[7]))
 #   config.set("architecture1", "delta_fc", str(delta[7]))

def set_bits_conf_LSTM(config,precisions,n_layers=2):
    # l0WX,L1WX,l0WH,L1WH,L0AX,L1AX,L0AH,L1AH
    xweight_bits_str=""
    xweight_act_str=""
    hweight_bits_str=""
    hweight_act_str=""
    for i in range(n_layers):
        xweight_bits_str = xweight_bits_str + str(precisions[i])
        if i<n_layers-1:
            xweight_bits_str = xweight_bits_str + ","
    for i in range(n_layers):
        xweight_act_str = xweight_act_str + str(precisions[i+2*n_layers])
        if i<n_layers-1:
            xweight_act_str = xweight_act_str + ","

    for i in range(n_layers):
        hweight_bits_str = hweight_bits_str + str(precisions[i+n_layers])
        if i<n_layers-1:
            hweight_bits_str = hweight_bits_str + ","
    for i in range(n_layers):
        hweight_act_str = hweight_act_str + str(precisions[i+3*n_layers])
        if i<n_layers-1:
            hweight_act_str = hweight_act_str + ","

    config.set("architecture1", "x_weights_bits", xweight_bits_str)
    config.set("architecture1", "x_weights_activation_bits",xweight_act_str)
    config.set("architecture1", "h_weights_bits", hweight_bits_str)
    config.set("architecture1", "h_weights_activation_bits", hweight_act_str)
    config.set("architecture1", "output_bits", str(precisions[n_layers*4+1]))
    config.set("architecture2", "number_of_bits", str(precisions[n_layers*4+1]))
    config.set("architecture2", "weights", str(precisions[n_layers*4]))

def set_bits_conf_GRU(config,precisions,n_layers=2):
    # l0WX,L1WX,l0WH,L1WH,L0AX,L1AX,L0AH,L1AH
    xweight_bits_str=""
    xweight_act_str=""
    hweight_bits_str=""
    hweight_act_str=""
    for i in range(n_layers):
        xweight_bits_str = xweight_bits_str + str(precisions[i])
        if i<n_layers-1:
            xweight_bits_str = xweight_bits_str + ","
    for i in range(n_layers):
        xweight_act_str = xweight_act_str + str(precisions[i+2*n_layers])
        if i<n_layers-1:
            xweight_act_str = xweight_act_str + ","

    for i in range(n_layers):
        hweight_bits_str = hweight_bits_str + str(precisions[i+n_layers])
        if i<n_layers-1:
            hweight_bits_str = hweight_bits_str + ","
    for i in range(n_layers):
        hweight_act_str = hweight_act_str + str(precisions[i+3*n_layers])
        if i<n_layers-1:
            hweight_act_str = hweight_act_str + ","

    config.set("architecture1", "x_weights_bits", xweight_bits_str)
    config.set("architecture1", "x_weights_activation_bits",xweight_act_str)
    config.set("architecture1", "h_weights_bits", hweight_bits_str)
    config.set("architecture1", "h_weights_activation_bits", hweight_act_str)
    config.set("architecture1", "output_bits", str(precisions[n_layers*4+1]))
    config.set("architecture2", "number_of_bits", str(precisions[n_layers*4+1]))
    config.set("architecture2", "weights", str(precisions[n_layers*4]))

def set_delta_conf_LSTM(config,delta,n_layers=2):
    delta_x_str=""
    delta_y_str=""
    for i in range(n_layers):
        delta_x_str=delta_x_str+str(delta[i])
        delta_y_str = delta_y_str + str(delta[i+n_layers])
        if i < n_layers - 1:
            delta_x_str = delta_x_str + ","
            delta_y_str = delta_y_str + ","
    config.set("architecture1", "delta_x",delta_x_str)
    config.set("architecture1", "delta_h", delta_y_str)

    config.set("architecture2", "delta", str(delta[n_layers*2]))
    config.set("architecture1", "delta_fc", str(delta[n_layers * 2]))

#Nesma This function changes in the configuration files to enable the use of the optimization dataset either for optimization or collecting statistics

def set_val_conf(kaldi_path,conf_main,config,outfolder):
    print('hello conf ')
    config.set("architecture1", "use_statistics", "False")
    config.set("architecture2", "use_statistics", "False")
    config.set("data_chunk", "fea",
               "fea_name=fbank \n fea_lst="+kaldi_path+"/egs/timit/s5/data/dev_o/feats.scp	\n fea_opts=apply-cmvn --utt2spk=ark:"+kaldi_path+"/egs/timit/s5/data/dev_o/utt2spk ark:"+kaldi_path+"/egs/timit/s5/fbank/cmvn_dev.ark ark:- ark:- | add-deltas --delta-order=0 ark:- ark:- |  \n cw_left=0 \n cw_right=0")
    config.set("data_chunk", "lab",
               " lab_name=lab_cd \n lab_folder="+kaldi_path+"/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali_dev \n lab_opts=ali-to-pdf \n lab_count_file=auto \n lab_data_folder="+kaldi_path+"/egs/timit/s5/data/dev_o/ \n lab_graph="+kaldi_path+"/egs/timit/s5/exp/tri3/graph \n \n \n lab_name=lab_mono \n lab_folder="+kaldi_path+"/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali_dev \n lab_opts=ali-to-phones --per-frame=true \n lab_count_file=none \n lab_data_folder="+kaldi_path+"/egs/timit/s5/data/dev_o/ \n lab_graph="+kaldi_path+"/egs/timit/s5/exp/tri3/graph")
    config.set("exp","out_info",outfolder+"/exp_files/forward_TIMIT_opt_ep23_ck0.info")

    conf_main.set("data_use", "forward_with", "TIMIT_opt")


def set_val_conf_1(kaldi_path,conf_main,config,outfolder):
    print('hello conf 1')
    config.set("architecture1", "use_statistics", "False")
    config.set("architecture2", "use_statistics", "False")
    config.set("data_chunk", "fea",
               "fea_name=fbank \n fea_lst="+kaldi_path+"/egs/timit/s5/data/dev_o1/feats.scp	\n fea_opts=apply-cmvn --utt2spk=ark:"+kaldi_path+"/egs/timit/s5/data/dev_o1/utt2spk ark:"+kaldi_path+"/egs/timit/s5/fbank/cmvn_dev.ark ark:- ark:- | add-deltas --delta-order=0 ark:- ark:- |  \n cw_left=0 \n cw_right=0")
    config.set("data_chunk", "lab",
               " lab_name=lab_cd \n lab_folder="+kaldi_path+"/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali_dev \n lab_opts=ali-to-pdf \n lab_count_file=auto \n lab_data_folder="+kaldi_path+"/egs/timit/s5/data/dev_o1/ \n lab_graph="+kaldi_path+"/egs/timit/s5/exp/tri3/graph \n \n \n lab_name=lab_mono \n lab_folder="+kaldi_path+"/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali_dev \n lab_opts=ali-to-phones --per-frame=true \n lab_count_file=none \n lab_data_folder="+kaldi_path+"/egs/timit/s5/data/dev_o1/ \n lab_graph="+kaldi_path+"/egs/timit/s5/exp/tri3/graph")
    config.set("exp","out_info",outfolder+"/exp_files/forward_TIMIT_opt_1_ep23_ck0.info")

    conf_main.set("data_use", "forward_with", "TIMIT_opt_1")

def set_val_conf_2(kaldi_path,conf_main,config,outfolder):
    print('hello conf 2')
    config.set("architecture1", "use_statistics", "False")
    config.set("architecture2", "use_statistics", "False")
    config.set("data_chunk", "fea",
               "fea_name=fbank \n fea_lst="+kaldi_path+"/egs/timit/s5/data/dev_o2/feats.scp	\n fea_opts=apply-cmvn --utt2spk=ark:"+kaldi_path+"/egs/timit/s5/data/dev_o2/utt2spk ark:"+kaldi_path+"/egs/timit/s5/fbank/cmvn_dev.ark ark:- ark:- | add-deltas --delta-order=0 ark:- ark:- |  \n cw_left=0 \n cw_right=0")
    config.set("data_chunk", "lab",
               " lab_name=lab_cd \n lab_folder="+kaldi_path+"/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali_dev \n lab_opts=ali-to-pdf \n lab_count_file=auto \n lab_data_folder="+kaldi_path+"/egs/timit/s5/data/dev_o2/ \n lab_graph="+kaldi_path+"/egs/timit/s5/exp/tri3/graph \n \n \n lab_name=lab_mono \n lab_folder="+kaldi_path+"/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali_dev \n lab_opts=ali-to-phones --per-frame=true \n lab_count_file=none \n lab_data_folder="+kaldi_path+"/egs/timit/s5/data/dev_o2/ \n lab_graph="+kaldi_path+"/egs/timit/s5/exp/tri3/graph")
    config.set("exp","out_info",outfolder+"/exp_files/forward_TIMIT_opt_2_ep23_ck0.info")

    conf_main.set("data_use", "forward_with", "TIMIT_opt_2")

def set_val_conf_3(kaldi_path,conf_main,config,outfolder):
    print('hello conf 3')
    config.set("architecture1", "use_statistics", "False")
    config.set("architecture2", "use_statistics", "False")
    config.set("data_chunk", "fea",
               "fea_name=fbank \n fea_lst="+kaldi_path+"/egs/timit/s5/data/dev_o3/feats.scp	\n fea_opts=apply-cmvn --utt2spk=ark:"+kaldi_path+"/egs/timit/s5/data/dev_o3/utt2spk ark:"+kaldi_path+"/egs/timit/s5/fbank/cmvn_dev.ark ark:- ark:- | add-deltas --delta-order=0 ark:- ark:- |  \n cw_left=0 \n cw_right=0")
    config.set("data_chunk", "lab",
               " lab_name=lab_cd \n lab_folder="+kaldi_path+"/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali_dev \n lab_opts=ali-to-pdf \n lab_count_file=auto \n lab_data_folder="+kaldi_path+"/egs/timit/s5/data/dev_o3/ \n lab_graph="+kaldi_path+"/egs/timit/s5/exp/tri3/graph \n \n \n lab_name=lab_mono \n lab_folder="+kaldi_path+"/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali_dev \n lab_opts=ali-to-phones --per-frame=true \n lab_count_file=none \n lab_data_folder="+kaldi_path+"/egs/timit/s5/data/dev_o3/ \n lab_graph="+kaldi_path+"/egs/timit/s5/exp/tri3/graph")
    config.set("exp","out_info",outfolder+"/exp_files/forward_TIMIT_opt_3_ep23_ck0.info")

    conf_main.set("data_use", "forward_with", "TIMIT_opt_3")

def set_val_conf_4(kaldi_path,conf_main,config,outfolder):

    config.set("architecture1", "use_statistics", "False")
    config.set("architecture2", "use_statistics", "False")
    config.set("data_chunk", "fea",
               "fea_name=fbank \n fea_lst="+kaldi_path+"/egs/timit/s5/data/dev_oo/feats.scp	\n fea_opts=apply-cmvn --utt2spk=ark:"+kaldi_path+"/egs/timit/s5/data/dev_oo/utt2spk ark:"+kaldi_path+"/egs/timit/s5/fbank/cmvn_dev.ark ark:- ark:- | add-deltas --delta-order=0 ark:- ark:- |  \n cw_left=0 \n cw_right=0")
    config.set("data_chunk", "lab",
               " lab_name=lab_cd \n lab_folder="+kaldi_path+"/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali_dev \n lab_opts=ali-to-pdf \n lab_count_file=auto \n lab_data_folder="+kaldi_path+"/egs/timit/s5/data/dev_oo/ \n lab_graph="+kaldi_path+"/egs/timit/s5/exp/tri3/graph \n \n \n lab_name=lab_mono \n lab_folder="+kaldi_path+"/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali_dev \n lab_opts=ali-to-phones --per-frame=true \n lab_count_file=none \n lab_data_folder="+kaldi_path+"/egs/timit/s5/data/dev_oo/ \n lab_graph="+kaldi_path+"/egs/timit/s5/exp/tri3/graph")
    config.set("exp","out_info",outfolder+"/exp_files/forward_TIMIT_opt_oo_ep23_ck0.info")

    conf_main.set("data_use", "forward_with", "TIMIT_opt_oo")

def set_val_conf_5(kaldi_path,conf_main,config,outfolder):

    config.set("architecture1", "use_statistics", "False")
    config.set("architecture2", "use_statistics", "False")
    config.set("data_chunk", "fea",
               "fea_name=fbank \n fea_lst="+kaldi_path+"/egs/timit/s5/data/dev/feats.scp	\n fea_opts=apply-cmvn --utt2spk=ark:"+kaldi_path+"/egs/timit/s5/data/dev/utt2spk ark:"+kaldi_path+"/egs/timit/s5/fbank/cmvn_dev.ark ark:- ark:- | add-deltas --delta-order=0 ark:- ark:- |  \n cw_left=0 \n cw_right=0")
    config.set("data_chunk", "lab",
               " lab_name=lab_cd \n lab_folder="+kaldi_path+"/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali_dev \n lab_opts=ali-to-pdf \n lab_count_file=auto \n lab_data_folder="+kaldi_path+"/egs/timit/s5/data/dev/ \n lab_graph="+kaldi_path+"/egs/timit/s5/exp/tri3/graph \n \n \n lab_name=lab_mono \n lab_folder="+kaldi_path+"/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali_dev \n lab_opts=ali-to-phones --per-frame=true \n lab_count_file=none \n lab_data_folder="+kaldi_path+"/egs/timit/s5/data/dev/ \n lab_graph="+kaldi_path+"/egs/timit/s5/exp/tri3/graph")
    config.set("exp","out_info",outfolder+"/exp_files/forward_TIMIT_dev_ep23_ck0.info")

    conf_main.set("data_use", "forward_with", "TIMIT_dev")

def set_val_conf_full(kaldi_path,conf_main,config,outfolder):
    config.set("architecture1", "use_statistics", "False")
    config.set("architecture2", "use_statistics", "False")
    config.set("data_chunk", "fea",
               "fea_name=fbank \n fea_lst="+kaldi_path+"/egs/timit/s5/data/train_o2/feats.scp	\n fea_opts=apply-cmvn --utt2spk=ark:"+kaldi_path+"/egs/timit/s5/data/train_o2/utt2spk ark:"+kaldi_path+"/egs/timit/s5/fbank/cmvn_train.ark ark:- ark:- | add-deltas --delta-order=0 ark:- ark:- |  \n cw_left=0 \n cw_right=0")
    config.set("data_chunk", "lab",
               " lab_name=lab_cd \n lab_folder="+kaldi_path+"/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali \n lab_opts=ali-to-pdf \n lab_count_file=auto \n lab_data_folder="+kaldi_path+"/egs/timit/s5/data/train_o2/ \n lab_graph="+kaldi_path+"/egs/timit/s5/exp/tri3/graph \n \n \n lab_name=lab_mono \n lab_folder="+kaldi_path+"/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali \n lab_opts=ali-to-phones --per-frame=true \n lab_count_file=none \n lab_data_folder="+kaldi_path+"/egs/timit/s5/data/train_o2/ \n lab_graph="+kaldi_path+"/egs/timit/s5/exp/tri3/graph")
    config.set("exp","out_info",outfolder+"/exp_files/forward_TIMIT_tr_op_ep23_ck0.info")

    conf_main.set("data_use", "forward_with", "TIMIT_tr_op")
#Nesma This function is changes in the configuration file to use the test set and use the collected statistics instead of computing the maximum values in the activations vectors
def set_test_conf(kaldi_path,conf_main,config,outfolder):

    config.set("architecture1", "use_statistics", "True")
    config.set("architecture2", "use_statistics", "True")

    config.set("data_chunk", "fea",
               "fea_name=fbank \n fea_lst="+kaldi_path+"/egs/timit/s5/data/test/feats.scp	\n fea_opts=apply-cmvn --utt2spk=ark:"+kaldi_path+"/egs/timit/s5/data/test/utt2spk ark:"+kaldi_path+"/egs/timit/s5/fbank/cmvn_test.ark ark:- ark:- | add-deltas --delta-order=0 ark:- ark:- |  \n cw_left=0 \n cw_right=0")
    config.set("data_chunk", "lab",
               " lab_name=lab_cd \n lab_folder="+kaldi_path+"/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali_test \n lab_opts=ali-to-pdf \n lab_count_file=auto \n lab_data_folder="+kaldi_path+"/egs/timit/s5/data/test \n lab_graph="+kaldi_path+"/egs/timit/s5/exp/tri3/graph \n \n \n lab_name=lab_mono \n lab_folder="+kaldi_path+"/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali_test \n lab_opts=ali-to-phones --per-frame=true \n lab_count_file=none \n lab_data_folder="+kaldi_path+"/egs/timit/s5/data/test \n lab_graph="+kaldi_path+"/egs/timit/s5/exp/tri3/graph")
    config.set("exp","out_info",outfolder+"/exp_files/forward_TIMIT_test_ep23_ck0.info")
    #TODO create validate for triaining in the conf file
    conf_main.set("data_use","forward_with","TIMIT_test")
def set_val_t_conf(kaldi_path,conf_main,config,outfolder):

    config.set("architecture1", "use_statistics", "True")
    config.set("architecture2", "use_statistics", "True")

    config.set("data_chunk", "fea",
               "fea_name=fbank \n fea_lst="+kaldi_path+"/egs/timit/s5/data/test_2/feats.scp	\n fea_opts=apply-cmvn --utt2spk=ark:"+kaldi_path+"/egs/timit/s5/data/test_2/utt2spk ark:"+kaldi_path+"/egs/timit/s5/fbank/cmvn_test_2.ark ark:- ark:- | add-deltas --delta-order=0 ark:- ark:- |  \n cw_left=0 \n cw_right=0")
    config.set("data_chunk", "lab",
               " lab_name=lab_cd \n lab_folder="+kaldi_path+"/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali_test_2 \n lab_opts=ali-to-pdf \n lab_count_file=auto \n lab_data_folder="+kaldi_path+"/egs/timit/s5/data/test_2 \n lab_graph="+kaldi_path+"/egs/timit/s5/exp/tri3/graph \n \n \n lab_name=lab_mono \n lab_folder="+kaldi_path+"/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali_test_2 \n lab_opts=ali-to-phones --per-frame=true \n lab_count_file=none \n lab_data_folder="+kaldi_path+"/egs/timit/s5/data/test_2 \n lab_graph="+kaldi_path+"/egs/timit/s5/exp/tri3/graph")
    config.set("exp","out_info",outfolder+"/exp_files/forward_TIMIT_opt_oo_ep23_ck0.info")
    #TODO create validate for triaining in the conf file
    conf_main.set("data_use","forward_with","TIMIT_opt_oo")


#Nesma
#This function run the quantization then the inference and return the error rate
# it takes the output folder relative path, the number of bits for weights and activations
# w0,w1,w2,w3 are the number of bits for weights of SRU
# iw0,iw1,iw2,iw3 are the number of bits for activations of sru
# p1,p2,p3 are the number of weights for the projection layers
# ip1,ip2,ip3 are the number of weights for the activations of the projection layers
# fc, fci are the number fo bits and activations for the fc layer
# if test is false, the inference run using the optimization data set, either for optimization or to collect statistics
# if test is false, the inference runs using the test data set and used the collected statistics (no get_vector_max is used)
def run_inference_for_optimization(kaldi_path,out_folder,precisions,delta=[],n_layers=4,test=False,opt_index=0,full=False):


    layers_bits =[]
    proj_bits=[16]
    for i in range(n_layers):
        layers_bits.append(precisions[i])
        if i<n_layers-1:
            proj_bits.append(precisions[2*n_layers+i])



    if len(sys.argv) > 1:
        cfg_file = sys.argv[1]
    else:
        if n_layers==4:
            cfg_file = "cfg/TIMIT_baselines/TIMIT_SRU_fbank.cfg"
        elif n_layers==6:
            cfg_file = "cfg/TIMIT_baselines/TIMIT_SRU_fbank_6.cfg"
    if not (os.path.exists(cfg_file)):
        sys.stderr.write("ERROR: The config file %s does not exist!\n" % (cfg_file))
        sys.exit(0)
    else:
        config = configparser.ConfigParser()
        config_main=configparser.ConfigParser()
        config.read(out_folder + '/quantize_TIMIT.cfg')
        config_main.read(cfg_file)
    #Nesma: set the activation bits in the configuration file. The weight bits are directly used during activation
    set_bits_conf(config, precisions,delta,n_layers)

    if test==True:
        set_test_conf(kaldi_path,config_main,config,out_folder)
    elif full == False:
        if opt_index==0:
            set_val_conf(kaldi_path,config_main, config,out_folder)
        elif opt_index==1:
            set_val_conf_1(kaldi_path, config_main, config, out_folder)
        elif opt_index==2:
            set_val_conf_2(kaldi_path, config_main, config, out_folder)
        elif opt_index==3:
            set_val_conf_3(kaldi_path, config_main, config, out_folder)
        elif opt_index==4:
            set_val_conf_4(kaldi_path, config_main, config, out_folder)
        else:
            set_val_conf_5(kaldi_path, config_main, config, out_folder)
    else:
        set_val_conf_full(kaldi_path, config_main, config, out_folder)
    with open(out_folder + '/quantize_TIMIT.cfg', 'w') as f:
        config.write(f)
    with open(cfg_file, 'w') as h:
        config_main.write(h)

    # quantize the sru model
    quantize(out_folder, 1, 'SRU', layers_bits,proj_bits,n_layers)
    # quantize the FC layer
    quantize(out_folder, 2, 'MLP', [precisions[len(precisions)-2]])
    error=float(run_inference(cfg_file))
    config.read(out_folder + '/quantize_TIMIT.cfg')
    sparsity=config.get("architecture1","sparsity")+","+config.get("architecture2","sparsity")

    return error,sparsity
def run_inference_for_optimization_GRU(kaldi_path,out_folder,precisions,delta=[0,0],n_layers=2,test=False,opt_index=0,full=False,skip=0):


    if len(sys.argv) > 1:
        cfg_file = sys.argv[1]
    else:
        cfg_file = "cfg/TIMIT_baselines/TIMIT_GRU_fbank.cfg"
    if not (os.path.exists(cfg_file)):
        sys.stderr.write("ERROR: The config file %s does not exist!\n" % (cfg_file))
        sys.exit(0)
    else:
        config = configparser.ConfigParser()
        config_main=configparser.ConfigParser()
        config.read(out_folder + '/quantize_TIMIT.cfg')
        config_main.read(cfg_file)
    #Nesma: set the activation bits in the configuration file. The weight bits are directly used during activation
    set_bits_conf_GRU(config,precisions,n_layers)
    set_delta_conf_LSTM(config, delta,n_layers)

    if test == True:
        set_test_conf(kaldi_path, config_main, config, out_folder)
    elif full == False:
        if   opt_index == 0:
            set_val_conf(  kaldi_path, config_main, config, out_folder)
        elif opt_index == 1:
            set_val_conf_1(kaldi_path, config_main, config, out_folder)
        elif opt_index == 2:
            set_val_conf_2(kaldi_path, config_main, config, out_folder)
        elif opt_index == 3:
            set_val_conf_3(kaldi_path, config_main, config, out_folder)
        elif opt_index == 4:
            set_val_conf_4(kaldi_path, config_main, config, out_folder)
        else:
            set_val_conf_5(kaldi_path, config_main, config, out_folder)
    else:
        set_val_conf_full(kaldi_path, config_main, config, out_folder)
    config.set("architecture1", "skip", str(skip))
    with open(out_folder + '/quantize_TIMIT.cfg', 'w') as f:
        config.write(f)
    with open(cfg_file, 'w') as h:
        config_main.write(h)

    # quantize the sru model
    quantize(out_folder, 1, 'GRU', precisions,n_layers=n_layers,skip=skip)
    # quantize the FC layer
    quantize(out_folder, 2, 'MLP', [precisions[n_layers*4]])

    error = float(run_inference(cfg_file))
    config.read(out_folder + '/quantize_TIMIT.cfg')
    sparsity = config.get("architecture1", "sparsity_x") +"," + config.get("architecture1", "sparsity_h") +"," + config.get("architecture2", "sparsity")
    print(sparsity)
    return error, sparsity
def run_inference_for_optimization_liGRU(kaldi_path,out_folder,precisions,delta=[0,0],n_layers=2,test=False,opt_index=0,full=False,skip=0):


    if len(sys.argv) > 1:
        cfg_file = sys.argv[1]
    else:
        cfg_file = "cfg/TIMIT_baselines/TIMIT_liGRU_fbank.cfg"
    if not (os.path.exists(cfg_file)):
        sys.stderr.write("ERROR: The config file %s does not exist!\n" % (cfg_file))
        sys.exit(0)
    else:
        config = configparser.ConfigParser()
        config_main=configparser.ConfigParser()
        config.read(out_folder + '/quantize_TIMIT.cfg')
        config_main.read(cfg_file)
    #Nesma: set the activation bits in the configuration file. The weight bits are directly used during activation
    set_bits_conf_GRU(config,precisions,n_layers)
    set_delta_conf_LSTM(config, delta,n_layers)

    if test == True:
        set_test_conf(kaldi_path, config_main, config, out_folder)
    elif full == False:
        if   opt_index == 0:
            set_val_conf(  kaldi_path, config_main, config, out_folder)
        elif opt_index == 1:
            set_val_conf_1(kaldi_path, config_main, config, out_folder)
        elif opt_index == 2:
            set_val_conf_2(kaldi_path, config_main, config, out_folder)
        elif opt_index == 3:
            set_val_conf_3(kaldi_path, config_main, config, out_folder)
        elif opt_index == 4:
            set_val_conf_4(kaldi_path, config_main, config, out_folder)
        else:
            set_val_conf_5(kaldi_path, config_main, config, out_folder)
    else:
        set_val_conf_full(kaldi_path, config_main, config, out_folder)
    config.set("architecture1", "skip", str(skip))
    with open(out_folder + '/quantize_TIMIT.cfg', 'w') as f:
        config.write(f)
    with open(cfg_file, 'w') as h:
        config_main.write(h)

    # quantize the sru model
    quantize(out_folder, 1, 'liGRU', precisions,n_layers=n_layers,skip=skip)
    # quantize the FC layer
    quantize(out_folder, 2, 'MLP', [precisions[n_layers*4]])

    error = float(run_inference(cfg_file))
    config.read(out_folder + '/quantize_TIMIT.cfg')
    sparsity = config.get("architecture1", "sparsity_x") +"," + config.get("architecture1", "sparsity_h") +"," + config.get("architecture2", "sparsity")
    print(sparsity)
    return error, sparsity

def run_inference_for_optimization_LSTM(kaldi_path,out_folder,precisions,delta=[0,0],n_layers=2,test=False,opt_index=0,full=False,skip=0):


    if len(sys.argv) > 1:
        cfg_file = sys.argv[1]
    else:
        cfg_file = "cfg/TIMIT_baselines/TIMIT_LSTM_fbank.cfg"
    if not (os.path.exists(cfg_file)):
        sys.stderr.write("ERROR: The config file %s does not exist!\n" % (cfg_file))
        sys.exit(0)
    else:
        config = configparser.ConfigParser()
        config_main=configparser.ConfigParser()
        config.read(out_folder + '/quantize_TIMIT.cfg')
        config_main.read(cfg_file)
    #Nesma: set the activation bits in the configuration file. The weight bits are directly used during activation
    set_bits_conf_LSTM(config,precisions,n_layers)
    set_delta_conf_LSTM(config, delta,n_layers)
    config.set("architecture1", "skip",str(skip))

    if test == True:
        set_test_conf(kaldi_path, config_main, config, out_folder)
    elif full == False:
        if   opt_index == 0:
            set_val_conf(  kaldi_path, config_main, config, out_folder)
        elif opt_index == 1:
            set_val_conf_1(kaldi_path, config_main, config, out_folder)
        elif opt_index == 2:
            set_val_conf_2(kaldi_path, config_main, config, out_folder)
        elif opt_index == 3:
            set_val_conf_3(kaldi_path, config_main, config, out_folder)
        elif opt_index == 4:
            set_val_conf_4(kaldi_path, config_main, config, out_folder)
        else:
            set_val_conf_5(kaldi_path, config_main, config, out_folder)
    else:
        set_val_conf_full(kaldi_path, config_main, config, out_folder)

    with open(out_folder + '/quantize_TIMIT.cfg', 'w') as f:
        config.write(f)
    with open(cfg_file, 'w') as h:
        config_main.write(h)

    # quantize the sru model
    quantize(out_folder, 1, 'LSTM', precisions,n_layers=n_layers,skip=skip)
    # quantize the FC layer
    quantize(out_folder, 2, 'MLP', [precisions[n_layers*4]])

    error = float(run_inference(cfg_file))
    config.read(out_folder + '/quantize_TIMIT.cfg')
    sparsity = config.get("architecture1", "sparsity_x") +"," + config.get("architecture1", "sparsity_h") +"," + config.get("architecture2", "sparsity")
    print(sparsity)
    return error, sparsity
def run_inference_for_optimization_lstm_delta(kaldi_path,out_folder,delta,test,full=False):

    if len(sys.argv) > 1:
        cfg_file = sys.argv[1]
    else:
        cfg_file = "cfg/TIMIT_baselines/TIMIT_LSTM_fbank.cfg"
    if not (os.path.exists(cfg_file)):
        sys.stderr.write("ERROR: The config file %s does not exist!\n" % (cfg_file))
        sys.exit(0)
    else:
        config = configparser.ConfigParser()
        config_main = configparser.ConfigParser()
        cfg_file_2=out_folder + '/quantize_TIMIT.cfg'
        config.read(cfg_file_2)
        config_main.read(cfg_file)
# Nesma: set the activation bits in the configuration file. The weight bits are directly used during activation
    set_delta_conf_LSTM(config, delta)

    if test == True:
        set_test_conf(kaldi_path, config_main, config, out_folder)
    else:
        set_val_conf_1(kaldi_path, config_main, config, out_folder)
       # set_val_conf(kaldi_path, config_main, config, out_folder)
    #else:
    #    set_val_conf_full(kaldi_path, config_main, config, out_folder)

    with open(out_folder + '/quantize_TIMIT.cfg', 'w') as f:
        config.write(f)
    with open(cfg_file, 'w') as h:
        config_main.write(h)

    # quantize the sru model
#    quantize(out_folder, 1, 'LSTM', layers_bits,2)
    # quantize the FC layer
# 
    #    WER1=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,2,2,2,2,8,2,2,2,2,2,2,2,16,16],[0,0,0,0,0,0,0,0,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
    #    quantize(out_folder, 2, 'MLP', [fc])
    error = float(run_inference(cfg_file))
    config.read(out_folder + '/quantize_TIMIT.cfg')
    sparsity = config.get("architecture1", "sparsity_x") +"," + config.get("architecture1", "sparsity_h") +"," + config.get("architecture2", "sparsity")

    return error, sparsity

#out_folder = "exp/TIMIT_SRU_fbank_z"
#quantize(out_folder, 1, 'SRU', [4,4,4,4],[16,16,16,16])
#quantize(out_folder, 2, 'MLP', [8])
#run_inference("cfg/TIMIT_baselines/TIMIT_LSTM_fbank.cfg")
#size=0.0036*32+0.17*32+0.17*32+0.17*32+0.08*32+0.08*32+0.08*32+0.08*32+0.15*32
percision=[8,2,2,2,16,2,2,2,16]
size=(23*550*4*percision[0] +1100*550*4*percision[1] +1100*550*4 *percision[2]+1100*550*4 *percision[3]+550 *550*4*percision[4]+550 *550*4*percision[5]+550 *550*4*percision[6]+550 *550*4*percision[7]+1904*1100*percision[8]+550*2*4*4*16)/8/1024/1024
print(size)
print(size/54.4*100)


#
#run_inference_for_optimization('/usr/local/home/nesrez/kaldi','exp/TIMIT_SRU_fbank_dnn_l6',[8,2,4,2,4,2,16,2,4,2,2,2,2,2,2,2,2,2,2,2,2,8,2,8],n_layers=6,test=False,full=True)
#run_inference_for_optimization('/usr/local/home/nesrez/kaldi','exp/TIMIT_SRU_fbank_dnn_l6',[8,2,4,2,4,2,16,2,4,2,2,2,2,2,2,2,2,2,2,2,2,8,2,8],n_layers=6,test=True)

#run_inference_for_optimization('/usr/local/home/nesrez/kaldi','exp/TIMIT_SRU_fbank_dnn_l6',[4,2,4,2,4,2,16,4,8,2,8,2,4,2,2,2,2,2,2,2,2,16,2,8],n_layers=6,test=False)
#run_inference_for_optimization('/usr/local/home/nesrez/kaldi','exp/TIMIT_SRU_fbank_dnn_l6',[4,2,4,2,4,2,16,4,8,2,8,2,4,2,2,2,2,2,2,2,2,16,2,8],n_layers=6,test=True)

#run_inference_for_optimization('/usr/local/home/nesrez/kaldi','exp/TIMIT_SRU_fbank_dnn_6',[8,4,4,4,16,4,8,16,4,2,4,16,16,8,4,16],n_layers=4,test=False)
#run_inference_for_optimization('/usr/local/home/nesrez/kaldi','exp/TIMIT_SRU_fbank_dnn_6',[4,2,4,4,16,4,16,16,8,4,4,16,16,16,2,16],n_layers=4,test=False,opt_index=1)
#run_inference_for_optimization('/usr/local/home/nesrez/kaldi','exp/TIMIT_SRU_fbank_dnn_6',[4,2,4,4,16,4,16,16,8,4,4,16,16,16,2,16],n_layers=4,test=False,opt_index=3)

#run_inference_for_optimization('/usr/local/home/nesrez/kaldi','exp/TIMIT_SRU_fbank_dnn_6',[8,4,4,4,16,4,8,16,4,2,4,16,16,8,4,16],n_layers=4,test=True)

#run_inference_for_optimization('/usr/local/home/nesrez/kaldi','exp/TIMIT_SRU_fbank_dnn_6',[4,2,4,2,16,2,8,2,2,2,2,2,2,8,2,4],n_layers=4,test=False,opt_index=2)
#run_inference_for_optimization('/usr/local/home/nesrez/kaldi','exp/TIMIT_SRU_fbank_dnn_6',[4,2,4,2,16,2,8,2,2,2,2,2,2,8,2,4],n_layers=4,test=False,opt_index=3)
#run_inference_for_optimization('/usr/local/home/nesrez/kaldi','exp/TIMIT_SRU_fbank_dnn_6',[4,2,4,2,16,2,8,2,2,2,2,2,2,8,2,4],n_layers=4,test=False,opt_index=1)
#run_inference_for_optimization('/usr/local/home/nesrez/kaldi','exp/TIMIT_SRU_fbank_dnn_6',[4,2,4,2,16,2,8,2,2,2,2,2,2,8,2,4],n_layers=4,test=False)
#run_inference_for_optimization('/usr/local/home/nesrez/kaldi','exp/TIMIT_SRU_fbank_dnn_6',[4,2,4,2,16,2,8,2,2,2,2,2,2,8,2,4],n_layers=4,test=True)



#percision=[8,2,2,2,2,16,2,2,2,2,16]
#size=(23*550*3*percision[0] +1100*550*3*percision[1] +1100*550*3 *percision[2]+1100*550*3 *percision[3]+1100*550*3*percision[4]+550 *550*2*percision[5]+550 *550*2*percision[6]+550 *550*2*percision[7]+550 *550*2*percision[8]+550 *550*2*percision[9]+1904*1100*percision[10]+550*550*5*16+550*2*3*5*16)/8/1024/1024
#print(size)
#print(size/53*100)


#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,8,8,8,8,8,8,8,8,8,8,8,16,16,16,16,16,16],[0,0,0,0,0,0,0,0,0,0,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2


#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,4,4,4,4,4,4,4,8,4,4,4,4,4,4,4,4,4],[0,0,0,0,0,0,0,0,0],4,False) #16 @ zer delta
#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,4,4,4,4,4,4,4,8,4,4,4,4,4,4,4,4,4],[0,0,0,0,0,0,0,0,0],4,True) #16.4 @ zer delta

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,4,4,4,4,8,4,4,4,4,4,4,4,4,4],[0,0,0,0,0,0,0,0,0],4,False) #16.3 @ zer delta
#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,4,4,4,4,8,4,4,4,4,4,4,4,4,4],[0,0,0,0,0,0,0,0,0],4,True) #16.3 @ zer delta

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,4,4,4,2,2,2,2,8,4,4,4,4,4,4,4,4,4],[0,0,0,0,0,0,0,0,0],4,False)#16.6 @ zer delta
#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,4,4,4,2,2,2,2,8,4,4,4,4,4,4,4,4,4],[0,0,0,0,0,0,0,0,0],4,True)#17 @ zer delta

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,4,4,4,2,2,2,2,8,4,4,4,4,4,4,4,2,8],[0,0,0,0,0,0,0,0,0],4,False) #17.6 @ zero delta
#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,4,4,4,2,2,2,2,8,4,4,4,4,4,4,4,2,8],[0,0,0,0,0,0,0,0,0],4,True)  #18.5   @ zer delta

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,2,2,2,2,8,4,4,4,4,4,4,4,4,4],[0,0,0,0,0,0,0,0,0],4,False) #16.2 @ zero delta
#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,2,2,2,2,8,4,4,4,4,4,4,4,4,4],[0,0,0,0,0,0,0,0,0],4,True)#17.1   @ zer delta

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[2,2,2,2,2,2,2,2,8,4,4,4,4,4,4,4,4,4],[0,0,0,0,0,0,0,0,0],4,False) #18.4 @ zer delta
#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[2,2,2,2,2,2,2,2,8,4,4,4,4,4,4,4,4,4],[0,0,0,0,0,0,0,0,0],4,True) #18.9 @ zer delta

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,4,4],[0,0,0,0,0,0,0,0,0],4,False) #17.1 @ zer delta
#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,4,4],[0,0,0,0,0,0,0,0,0],4,True) #17.8 @ zer delta








# WER1=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,0,0,0,0,0,0,0,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# WER2=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,0,0,0,0,0,0,0,0],4,False,1) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# WER3=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,0,0,0,0,0,0,0,0],4,False,2) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# WER4=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,0,0,0,0,0,0,0,0],4,False,3) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# print(max(WER1,WER2,WER3,WER4))
# run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,0,0,0,0,0,0,0,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# WER1=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,0,0,0,0,0,0,0,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# WER2=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,0,0,0,0,0,0,0,0],4,False,1) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# WER3=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,0,0,0,0,0,0,0,0],4,False,2) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# WER4=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,0,0,0,0,0,0,0,0],4,False,3) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# print(max(WER1,WER2,WER3,WER4))
# run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,0,0,0,0,0,0,0,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# WER1=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,16,16,16,0,16,16,16,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# WER2=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,16,16,16,0,16,16,16,0],4,False,1) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# WER3=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,16,16,16,0,16,16,16,0],4,False,2) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# WER4=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,16,16,16,0,16,16,16,0],4,False,3) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# print(max(WER1,WER2,WER3,WER4))
#
# run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,16,16,16,0,16,16,16,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# WER1=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,16,16,16,0,0,0,0,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# WER2=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,16,16,16,0,0,0,0,0],4,False,1) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# WER3=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,16,16,16,0,0,0,0,0],4,False,2) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# WER4=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,16,16,16,0,0,0,0,0],4,False,3) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# print(max(WER1,WER2,WER3,WER4))
#
# run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,16,16,16,0,0,0,0,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# WER1=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,0,0,0,0,16,16,16,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# WER2=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,0,0,0,0,16,16,16,0],4,False,1) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# WER3=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,0,0,0,0,16,16,16,0],4,False,2) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# WER4=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,0,0,0,0,16,16,16,0],4,False,3) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# print(max(WER1,WER2,WER3,WER4))
#
# run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,0,0,0,0,16,16,16,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# WER1=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,32,32,32,0,16,16,16,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# WER2=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,32,32,32,0,16,16,16,0],4,False,1) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# WER3=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,32,32,32,0,16,16,16,0],4,False,2) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# WER4=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,32,32,32,0,16,16,16,0],4,False,3) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# print(max(WER1,WER2,WER3,WER4))
#
# run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,32,32,32,0,16,16,16,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# WER1=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,16,16,16,0,32,32,32,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# WER2=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,16,16,16,0,32,32,32,0],4,False,1) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# WER3=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,16,16,16,0,32,32,32,0],4,False,2) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# WER4=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,16,16,16,0,32,32,32,0],4,False,3) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# print(max(WER1,WER2,WER3,WER4))
#
# run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,16,16,16,0,32,32,32,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# WER1=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,32,32,32,0,0,0,0,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# WER2=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,32,32,32,0,0,0,0,0],4,False,1) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# WER3=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,32,32,32,0,0,0,0,0],4,False,2) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# WER4=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,32,32,32,0,0,0,0,0],4,False,3) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# print(max(WER1,WER2,WER3,WER4))
#
# run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,32,32,32,0,0,0,0,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# WER1=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,0,0,0,0,32,32,32,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# WER2=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,0,0,0,0,32,32,32,0],4,False,1) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# WER3=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,0,0,0,0,32,32,32,0],4,False,2) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# WER4=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,0,0,0,0,32,32,32,0],4,False,3) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# print(max(WER1,WER2,WER3,WER4))
#
# run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,0,0,0,0,32,32,32,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
#
# WER1=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,32,32,32,0,32,32,32,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# WER2=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,32,32,32,0,32,32,32,0],4,False,1) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# WER3=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,32,32,32,0,32,32,32,0],4,False,2) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# WER4=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,32,32,32,0,32,32,32,0],4,False,3) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#
# print(max(WER1,WER2,WER3,WER4))
#
# run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,32,32,32,0,32,32,32,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2


#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,3,3,3,0,1,1,1,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,3,3,3,0,1,1,1,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2


#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,1,1,1,0,3,3,3,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,1,1,1,0,3,3,3,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2


#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,3,3,3,0,0,0,0,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,3,3,3,0,0,0,0,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,0,0,0,0,3,3,3,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,0,0,0,0,3,3,3,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,2,2,2,0,3,3,3,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,2,2,2,0,3,3,3,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,3,3,3,0,2,2,2,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,3,3,3,0,2,2,2,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,3,3,3,0,3,3,3,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,3,3,3,0,3,3,3,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,4,4,4,0,0,0,0,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,4,4,4,0,0,0,0,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,0,0,0,0,4,4,4,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,0,0,0,0,4,4,4,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,4,4,4,0,1,1,1,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,4,4,4,0,1,1,1,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2


#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,1,1,1,0,4,4,4,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,1,1,1,0,4,4,4,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2


#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,2,2,2,0,4,4,4,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,2,2,2,0,4,4,4,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,4,4,4,0,2,2,2,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,4,4,4,0,2,2,2,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,2,2,2,0,4,4,4,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,2,2,2,0,4,4,4,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,4,4,4,0,3,3,3,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,4,4,4,0,3,3,3,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,3,3,3,0,4,4,4,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,3,3,3,0,4,4,4,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,4,4,4,0,4,4,4,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,4,4,4,0,4,4,4,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2



#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,5,5,5,0,0,0,0,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,5,5,5,0,0,0,0,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,0,0,0,0,5,5,5,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,0,0,0,0,5,5,5,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,5,5,5,0,1,1,1,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,5,5,5,0,1,1,1,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2


#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,1,1,1,0,5,5,5,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,1,1,1,0,5,5,5,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2


#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,2,2,2,0,5,5,5,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,2,2,2,0,5,5,5,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,5,5,5,0,2,2,2,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,5,5,5,0,2,2,2,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,2,2,2,0,5,5,5,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,2,2,2,0,5,5,5,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,5,5,5,0,3,3,3,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,5,5,5,0,3,3,3,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,3,3,3,0,5,5,5,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,3,3,3,0,5,5,5,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,5,5,5,0,4,4,4,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,5,5,5,0,4,4,4,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,4,4,4,0,5,5,5,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,4,4,4,0,5,5,5,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,5,5,5,0,5,5,5,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,8,8,8,16,8,8,8,16,16],[0,5,5,5,0,5,5,5,0],4,True) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2




#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,4,4,4,16,4,4,4,16,16],[0,0,0,0,0,0,0,0,0],4,True) #15.3 @ zer delta,  [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 15.2
#print((0.0036*8+0.17*2+0.17*2+0.17*2+0.08*16+0.08*2+0.08*2+0.08*2+0.15*16)/size)
#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,4,4,4,16,4,4,4,16,16],[0,0.2,0.2,0.2,0,0.2,0.2,0.2,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,4,4,4,16,4,4,4,16,16],[0,0.1,0.1,0.1,0,0.1,0.1,0.1,0],4,True) #15.3 @ zer delta,  [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 15.2
#print((0.0036*8+0.17*2+0.17*2+0.17*2+0.08*16+0.08*2+0.08*2+0.08*2+0.15*16)/size)


#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,16,16,16,16,16,16,16,4,4],[0,0,0,0,0,0,0,0,0],4,False) #14.6 @ zer delta
#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,16,16,16,16,16,16,16,4,4],[0,0,0,0,0,0,0,0,0],4,True) #15.2 @ zer delta
#print((0.0036*8+0.17*2+0.17*2+0.17*2+0.08*16+0.08*2+0.08*2+0.08*2+0.15*4)/size)

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,16,16,16,16,4,4,4,16,16],[0,0,0,0,0,0,0,0,0],4,False) #14.6 @ zer delta
#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,16,16,16,16,4,4,4,16,16],[0,0,0,0,0,0,0,0,0],4,True) #15 @ zer delta
#print((0.0036*8+0.17*2+0.17*2+0.17*2+0.08*16+0.08*2+0.08*2+0.08*2+0.15*16)/size)

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,2,2,2,2,8,4,2,4,2,2,2,2,2,4],[0,0,0,0,0,0,0,0,0],4,False,skip=1)#14.7 @ zer delta
#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,2,2,2,2,8,4,2,4,2,2,2,2,2,4],[0,0,0,0,0,0,0,0,0],4,True,skip=1)#15.2 @ zer delta
#print((0.0036*8+0.17*2+0.17*2+0.17*2+0.08*16+0.08*2+0.08*2+0.08*2+0.15*16)/size)  #16.6 with skip=1




#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,4,4,4,16,16,16,16,2,16],[0,0,0,0,0,0,0,0,0],4,False) #15.8 @ zero delta
#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,4,4,4,16,16,16,16,2,16],[0,0,0,0,0,0,0,0,0],4,True)  # 16.5  @ zer delta

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/retraining_lstm/retrain_rrx13',[8,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,2,16],[0,0,0,0,0,0,0,0,0],4,False) #15.3 @ zero delta
#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/retraining_lstm/retrain_rrx13',[8,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,2,16],[0,0,0,0,0,0,0,0,0],4,True)# 17  @ zer delta

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,2,2,2,2,16,16,16,16,16,16,16,16,16,16],[0,0,0,0,0,0,0,0,0],4,False) #15.7 @ zer delta
#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,2,2,2,2,16,16,16,16,16,16,16,16,16,16],[0,0,0,0,0,0,0,0,0],4,True) #16.8 @ zer delta

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,4,4,4,4,2,2,2,8,4,4,4,4,4,4,4,4,4],[0,0,0,0,0,0,0,0,0],4,False) #16.3 @ zer delta
#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,4,4,4,4,2,2,2,8,4,4,4,4,4,4,4,4,4],[0,0,0,0,0,0,0,0,0],4,True) #16.7 @ zer delta

#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[16, 4,4,2, 16,16,4,16,8,8,16,16,16,8,2,4, 8,4],[0,0,0,0,0,0,0,0,0],4,True)



#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,4,4,4,16,4,4,4,16,16],[0,0.2,0.2,0.2,0,0.2,0.2,0.2,0],4,False) #14.7 @ zer delta , [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 14.2
#run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',[8,2,2,2,16,2,2,2,16,4,4,4,16,4,4,4,16,16],[0,0.1,0.1,0.1,0,0.1,0.1,0.1,0],4,True) #15.3 @ zer delta,  [0,0.1,0.1,0.1,0,0.1,0.1,0.1,0] 15.2
#print((0.0036*8+0.17*2+0.17*2+0.17*2+0.08*16+0.08*2+0.08*2+0.08*2+0.15*16)/size)


#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,16,16,16,16,16,16,16,16,16,4,4],[0,0,0,0,0,0,0,0,0,0,0],5,False) #14.1 @ zer delta
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,16,16,16,16,16,16,16,16,16,4,4],[0,0,0,0,0,0,0,0,0,0,0],5,True) #15.6 @ zer delta

#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,16,16,16,16,16,16,16,16,16,16,16],[0,0,0,0,0,0,0,0,0,0,0],5,False) #14 @ zer delta
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,16,16,16,16,16,16,16,16,16,16,16],[0,0,0,0,0,0,0,0,0,0,0],5,True) #15.6 @ zer delta

#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,4,4,4,4,16,16,16,16,16,16,16],[0,0,0,0,0,0,0,0,0,0,0],5,False)#14 @ zer delta
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,4,4,4,4,16,16,16,16,16,16,16],[0,0,0,0,0,0,0,0,0,0,0],5,True)#15.5 @ zer delta


#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,4,4,4,4,16,16,16,16,16,2,16],[0,0,0,0,0,0,0,0,0,0,0],5,False) #14 @ zero delta
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,4,4,4,4,16,16,16,16,16,2,16],[0,0,0,0,0,0,0,0,0,0,0],5,True)  #15.8  @ zer delta

#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[2,2,2,2,2,16,2,2,2,2,16,16,16,16,16,16,16,16,16,16,16,16],[0,0,0,0,0,0,0,0,0,0,0],5,False) # @ zero delta
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[2,2,2,2,2,16,2,2,2,2,16,16,16,16,16,16,16,16,16,16,16,16],[0,0,0,0,0,0,0,0,0,0,0],5,True)#   @ zer delta

#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,2,2,2,2,2,16,16,16,16,16,16,16,16,16,16,16,16],[0,0,0,0,0,0,0,0,0,0,0],5,False) #14.1 @ zer delta
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,2,2,2,2,2,16,16,16,16,16,16,16,16,16,16,16,16],[0,0,0,0,0,0,0,0,0,0,0],5,True) #15.5 @ zer delta

#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,4,4,4,4,4,2,2,2,2,8,4,4,4,4,4,4,4,4,4,4,4],[0,0,0,0,0,0,0,0,0,0,0],5,False) # @ zer delta
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,4,4,4,4,4,2,2,2,2,8,4,4,4,4,4,4,4,4,4,4,4],[0,0,0,0,0,0,0,0,0,0,0],5,True) #15.5 @ zer delta


#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,4,4,4,4,4,4,4,4,4,8,4,4,4,4,4,4,4,4,4,4,4],[0,0,0,0,0,0,0,0,0,0,0],5,False) #14.4 @ zer delta
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,4,4,4,4,4,4,4,4,4,8,4,4,4,4,4,4,4,4,4,4,4],[0,0,0,0,0,0,0,0,0,0,0],5,True) #15.6 @ zer delta

#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,4,4,4,4,4,8,4,4,4,4,4,4,4,4,4,4,4],[0,0,0,0,0,0,0,0,0,0,0],5,False) # @ zer delta
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,4,4,4,4,4,8,4,4,4,4,4,4,4,4,4,4,4],[0,0,0,0,0,0,0,0,0,0,0],5,True) # @ zer delta

#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,4,4,4,4,2,2,2,2,2,8,4,4,4,4,4,4,4,4,4,4,4],[0,0,0,0,0,0,0,0,0,0,0],5,False)# @ zer delta
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,4,4,4,4,2,2,2,2,2,8,4,4,4,4,4,4,4,4,4,4,4],[0,0,0,0,0,0,0,0,0,0,0],5,True)#15.6 @ zer delta

#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,4,4,4,4,2,2,2,2,2,8,4,4,4,4,4,4,4,4,4,2,8],[0,0,0,0,0,0,0,0,0,0,0],5,False) # @ zero delta
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,4,4,4,4,2,2,2,2,2,8,4,4,4,4,4,4,4,4,4,2,8],[0,0,0,0,0,0,0,0,0,0,0],5,True)  #15.7   @ zer delta

#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,2,2,2,2,2,8,4,4,4,4,4,4,4,4,4,4,4],[0,0,0,0,0,0,0,0,0,0,0],5,False) # @ zero delta
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,2,2,2,2,2,8,4,4,4,4,4,4,4,4,4,4,4],[0,0,0,0,0,0,0,0,0,0,0],5,True)#15.5   @ zer delta

#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[2,2,2,2,2,2,2,2,2,2,8,4,4,4,4,4,4,4,4,4,4,4],[0,0,0,0,0,0,0,0,0,0,0],5,False) # @ zer delta
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[2,2,2,2,2,2,2,2,2,2,8,4,4,4,4,4,4,4,4,4,4,4],[0,0,0,0,0,0,0,0,0,0,0],5,True) #16.9 @ zer delta

#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,4,4,4,4],[0,0,0,0,0,0,0,0,0,0,0],5,False) # @ zer delta
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,4,4,4,4],[0,0,0,0,0,0,0,0,0,0,0],5,True) #15.9 @ zer delta

#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,4,4,2,4],[0,0,0,0,0,0,0,0,0,0,0],5,False) # @ zer delta
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,4,4,2,4],[0,0,0,0,0,0,0,0,0,0,0],5,True) #15.9 @ zer delta

#delta experiments on GRU
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,4,4,4,4,16,4,4,4,4,16,16],[0,0,0,0,0,0,0,0,0,0,0,0],5,False) # @ zer delta
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,4,4,4,4,16,4,4,4,4,16,16],[0,0,0,0,0,0,0,0,0,0,0,0],5,True) #15.5 @ zer delta
#97.29505920410156,56.182334899902344,51.07338333129883,50.45367431640625,50.40891647338867,97.2727279663086,43.45454406738281,45.3636360168457,47.272727966308594,27.90909194946289,100

#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,4,4,4,4,16,4,4,4,4,16,16],[0,0,0,0,0,0,0.125,0.125,0.125,0.125,1,0],5,False) # @ zer delta
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,4,4,4,4,16,4,4,4,4,16,16],[0,0,0,0,0,0,0.125,0.125,0.125,0.125,1,0],5,True) # @ zer delta
#97.29505920410156,56.182334899902344,51.02394104003906,50.41828918457031,50.365196228027344,97.2727279663086,21.090909957885742,21.545454025268555,25.454544067382812,16.363637924194336,100
#15.6
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,4,4,4,4,16,4,4,4,4,16,16],[0,0.125,0.125,0.125,0.125,0,0,0,0,0,0,0],5,False) # @ zer delta
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,4,4,4,4,16,4,4,4,4,16,16],[0,0.125,0.125,0.125,0.125,0,0,0,0,0,0,0],5,True) # @ zer delta

#97.29505920410156,30.914644241333008,27.293373107910156,27.63861656188965,28.003814697265625,97.2727279663086,42.272727966308594,40.818180084228516,45.3636360168457,27.636363983154297,100
#16
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,4,4,4,4,16,4,4,4,4,16,16],[0,0.125,0.125,0.125,0.125,0,0.125,0.125,0.125,0.125,0],5,False) # @ zer delta
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,4,4,4,4,16,4,4,4,4,16,16],[0,0.125,0.125,0.125,0.125,0,0.125,0.125,0.125,0.125,0],5,True) # @ zer delta
#97.29505920410156,30.914644241333008,27.351144790649414,27.545455932617188,27.942228317260742,97.2727279663086,22.454544067382812,21.363636016845703,21.727272033691406,15.818181991577148,100
#15.8
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,4,4,4,4,16,4,4,4,4,16,16],[0,0,0,0,0,0,0.25,0.25,0.25,0.25,0],5,False) # @ zer delta
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,4,4,4,4,16,4,4,4,4,16,16],[0,0,0,0,0,0,0.25,0.25,0.25,0.25,0],5,True) # @ zer delta
#97.29505920410156,56.182334899902344,51.08466339111328,50.3702278137207,50.290252685546875,97.2727279663086,13.272727966308594,13.727272987365723,14.90909194946289,10.090909004211426,100
#15.8
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,4,4,4,4,16,4,4,4,4,16,16],[0,0.25,0.25,0.25,0.25,0,0,0,0,0,0,0],5,False) # @ zer delta
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,4,4,4,4,16,4,4,4,4,16,16],[0,0.25,0.25,0.25,0.25,0,0,0,0,0,0],5,True) # @ zer delta
#97.29505920410156,19.674530029296875,17.24757194519043,17.928869247436523,18.187196731567383,97.2727279663086,40.90909194946289,39.181819915771484,41.818180084228516,26.09090805053711,100
#16.8
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,4,4,4,4,16,4,4,4,4,16,16],[0,0.25,0.25,0.25,0.25,0,0.25,0.25,0.25,0.25,0],5,False) # @ zer delta
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,4,4,4,4,16,4,4,4,4,16,16],[0,0.25,0.25,0.25,0.25,0,0.25,0.25,0.25,0.25,0],5,True) # @ zer delta
#97.29505920410156,30.914644241333008,27.32928466796875,27.52654266357422,27.846981048583984,97.2727279663086,12.727272033691406,11.545454978942871,12.909090042114258,9.727272987365723,100
#16.2
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,4,4,4,4,16,4,4,4,4,16,16],[0,0.25,0.25,0.25,0.25,0,0.25,0.25,0.25,0.25,0],5,False) # @ zer delta
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,4,4,4,4,16,4,4,4,4,16,16],[0,0.25,0.25,0.25,0.25,0,0.25,0.25,0.25,0.25,0],5,True) # @ zer delta
#97.29505920410156,19.674530029296875,17.255029678344727,17.95749282836914,18.28643226623535,97.2727279663086,20.0,19.18181800842285,18.636363983154297,13.36363697052002,100
#17.1
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,4,4,4,4,16,4,4,4,4,16,16],[0,0.25,0.25,0.25,0.25,0,0.25,0.25,0.25,0.25,0],5,False) # @ zer delta
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,4,4,4,4,16,4,4,4,4,16,16],[0,0.25,0.25,0.25,0.25,0,0.25,0.25,0.25,0.25,0],5,True) # @ zer delta
#17.2
#97.29505920410156,19.674530029296875,17.228487014770508,17.866758346557617,18.14156723022461,97.2727279663086,11.727272987365723,11.090909004211426,10.727272987365723,8.636363983154297,100


#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,8,8,8,8,16,8,8,8,8,16,16],[0,0,0,0,0,0,0,0,0,0,0,0],5,False) # @ zer delta
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,8,8,8,8,16,8,8,8,8,16,16],[0,0,0,0,0,0,0,0,0,0,0,0],5,True) #15.5 @ zer delta

#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,8,8,8,8,16,8,8,8,8,16,16],[0,0,0,0,0,0,0.25,0.25,0.25,0.25,1,0],5,False) # @ zer delta
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,8,8,8,8,16,8,8,8,8,16,16],[0,0,0,0,0,0,0.25,0.25,0.25,0.25,0],5,True) # @ zer delta
#15.7
#97.29505920410156,91.91741943359375,89.6663818359375,88.44517517089844,89.64173889160156,97.2727279663086,17.363636016845703,16.90909194946289,18.545454025268555,13.909090995788574,100

#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,8,8,8,8,16,8,8,8,8,16,16],[0,0.25,0.25,0.25,0.25,0,0,0,0,0,0,0],5,False) # @ zer delta
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,8,8,8,8,16,8,8,8,8,16,16],[0,0.25,0.25,0.25,0.25,0,0,0,0,0,0,0],5,True) # @ zer delta
#16.1
#97.29505920410156,25.384628295898438,22.4559326171875,23.007980346679688,23.554302215576172,97.2727279663086,83.81818389892578,82.90908813476562,88.2727279663086,58.909088134765625,100

#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,8,8,8,8,16,8,8,8,8,16,16],[0,0.25,0.25,0.25,0.25,0,0.25,0.25,0.25,0.25,0],5,False) # @ zer delta
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,8,8,8,8,16,8,8,8,8,16,16],[0,0.25,0.25,0.25,0.25,0,0.25,0.25,0.25,0.25,0],5,True) # @ zer delta
#16.3
#97.29505920410156,25.384628295898438,22.445871353149414,22.981782913208008,23.480567932128906,97.2727279663086,17.272727966308594,14.63636302947998,15.272727966308594,11.636363983154297,100

#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,8,8,8,8,16,8,8,8,8,16,16],[0,0,0,0,0,0,0,0.5,0.5,0.5,0.5,0],5,False) # @ zer delta
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,8,8,8,8,16,8,8,8,8,16,16],[0,0,0,0,0,0,0.5,0.5,0.5,0.5,0],5,True) # @ zer delta
#15.7
#97.29505920410156,91.91741943359375,89.60929870605469,88.36658477783203,89.70193481445312,97.2727279663086,7.909090995788574,7.272727012634277,6.818181991577148,6.727272987365723,100

#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,8,8,8,8,16,8,8,8,8,16,16],[0,0.5,0.5,0.5,0.5,0,0,0,0,0,0,0],5,False) # @ zer delta
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,8,8,8,8,16,8,8,8,8,16,16],[0,0.5,0.5,0.5,0.5,0,0,0,0,0,0],5,True) # @ zer delta
#97.29505920410156,12.209576606750488,10.783309936523438,11.256592750549316,11.518563270568848,97.2727279663086,82.63636016845703,81.09091186523438,84.54545593261719,56.818180084228516,100
#20.4
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,8,8,8,8,16,8,8,8,8,16,16],[0,0.25,0.25,0.25,0.25,0,0.5,0.5,0.5,0.5,0],5,False) # @ zer delta
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,8,8,8,8,16,8,8,8,8,16,16],[0,0.25,0.25,0.25,0.25,0,0.5,0.5,0.5,0.5,0],5,True) # @ zer delta
#16.8
#97.29505920410156,25.384628295898438,22.513877868652344,22.93563461303711,23.366931915283203,97.2727279663086,7.0,5.727272987365723,5.2727274894714355,5.727272987365723,100

#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,8,8,8,8,16,8,8,8,8,16,16],[0,0.5,0.5,0.5,0.5,0,0.25,0.25,0.25,0.25,0],5,False) # @ zer delta
#run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_GRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,8,8,8,8,16,8,8,8,8,16,16],[0,0.5,0.5,0.5,0.5,0,0.25,0.25,0.25,0.25,0],5,True) # @ zer delta
#20.8
#97.29505920410156,12.209576606750488,10.76474666595459,11.265613555908203,11.503990173339844,97.2727279663086,14.818182945251465,12.272727966308594,11.818181991577148,12.181818008422852,100

#run_inference_for_optimization_liGRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_liGRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,8,8,8,8,16,8,8,8,8,16,16],[0,0.5,0.5,0.5,0.5,0,0.5,0.5,0.5,0.5,0],5,False) # @ zer delta
#run_inference_for_optimization_liGRU('/usr/local/home/nesrez/kaldi','exp/TIMIT_liGRU_fbank',[8,2,2,2,2,16,2,2,2,2,16,8,8,8,8,16,8,8,8,8,16,16],[0,0.5,0.5,0.5,0.5,0,0.5,0.5,0.5,0.5,0],5,True) # @ zer delta
exp_gru_new=[#[16,16,16,16,16,  16,16,16,16,16,  16,16,16,16,16,  16,16,16,16,16, 16,16],
     #[8,8,8,8,8, 8,8,8,8,8, 8,8,8,8,8,  8,8,8,8,8,  8,8],   #14.8,15
     #[8,4,4,4,4,   4,4,4,4,4,   8,4,4,4,4,    4,4,4,4,4,  4,4],
#[8,2,2,2,2,   2,2,2,2,2,  8,8,8,8, 8,    8,8,8,8,8,  8,8],#M1
# [8,2,2,2,2,   2,2,2,2,2,  8,8,8,8,8,     8,8,8,8,8,  4,4],#M2
# [8,2,2,2,2,   2,2,2,2,2,  8,8,8,8,8,     8,8,8,8,8,  2,8],#M3
# [2,2,2,2,2,   2,2,2,2,2,  8,8,8,8,8,     8,8,8,8,8,  2,8],#M4
    # [8,2,2,2,2,   16,2,2,2,2,  16,16,16,16,16,    16,16,16,16,16, 4,4], #3
    # [8,2,2,2,2,   16,2,2,2,2,  16,16,16,16,16,    16,4,4,4,4,   16,16],
    # [8,2,2,2,2,   4,4,4,4,4,   8,4,4,4,4,  4,4,4,4,4, 4,4],
    # [8,4,4,4,4,   4,4,4,4,4,   8,4,4,4,4,  4,4,4,4,4, 4,4],
    # [8,2,2,2,2,   16,2,2,2,2,   16,4,4,4,4,  16,16,16,16,16, 2,16],
    # [8,4,4,4,4,   4,2,2,2,2,   8,4,4,4,4,  4,4,4,4,4, 4,4],
    # [8,2,2,2,2,   2,2,2,2,2,   16,4,4,4,4,  4,4,4,4,4, 2,4],
    # [2,2,2,2,2,   2,2,2,2,2,   16,16,16,16,16,  16,16,16,16,16, 16,16],
    # [8,4,4,4,4,   2,2,2,2,2,   8,4,4,4,4,  4,4,4,4,4, 4,4],
    # [8,2,2,2,2,   2,2,2,2,2,   8,4,4,4,4,  4,4,4,4,4, 4,4],
    # [8,2,2,2,2,   2,2,2,2,2,   4,4,4,4,4,  4,4,4,4,4, 4,4]
]
exp_lstm=[[16,16,16,16,16,  16,16,16,16,16,  16,16,16,16,16,  16,16,16,16,16, 16,16],
    #[8,8,8,8,8, 8,8,8,8,8, 8,8,8,8,8,  8,8,8,8,8,  8,8],   #14.8,15
     [8,4,4,4,   4,4,4,4,   8,4,4,4,    4,4,4,4,  4,4],
     [8,2,2,2,    2,2,2,2,   8,4,4,4,   4,4,4,4, 4, 4],

     [8,2,2,2,   16,2,2,2,  16,16,16,16,    16,16,16,16, 4,4],
     [8,2,2,2,   16,2,2,2,  16,16,16,16,    16,4,4,4,   16,16],
     [8,2,2,2,   4,4,4,4,   8,4,4,4,  4,4,4,4, 4,4],
     [8,4,4,4,   4,4,4,4,   8,4,4,4,  4,4,4,4, 4,4],
     [8,2,2,2,   16,2,2,2,   16,4,4,4,  16,16,16,16, 2,16],
     [8,4,4,4,   4,2,2,2,   8,4,4,4,  4,4,4,4, 4,4],
     [8,2,2,2,   2,2,2,2,   16,4,4,4,  4,4,4,4, 2,4],
     [2,2,2,2,   2,2,2,2,   16,16,16,16,  16,16,16,16, 16,16],
     [8,4,4,4,   2,2,2,2,   8,4,4,4,  4,4,4,4, 4,4],
     [8,2,2,2,   2,2,2,2,   8,4,4,4,  4,4,4,4, 4,4],
     [8,2,2,2,   2,2,2,2,   4,4,4,4,  4,4,4,4, 4,4]
]
exp_lstm_new1=[

  #[16,16,16,16,  16,16,16,16, 16,16,16,16, 16,16,16,16,   16,16],
  [8,8,8,8, 8,8,8,8, 8,8,8,8,  8,8,8,8,  8,8],
  #[4,4,4,4, 4,4,4,4, 4,4,4,4,  4,4,4,4, 4,4],
  #[2,2,2,2, 2,2,2,2, 2,2,2,2,  2,2,2,2, 2,4],
# [16,2,2,2,   16,2,2,2,  16,16,16,16,    16,16,16,16, 16,16],#M1
# [16,2,2,2,   16,2,2,2,  16,16,16,16,    16,16,16,16, 4,4],#M2
# [16,2,2,2,   16,2,2,2,  16,4,4,4,   16,16,16,16,  16,16],#M3
# [16,2,2,2,   16,2,2,2,  16,16,16,16,   16,4,4,4,  16,16],#M4
# [16,2,2,2,   2,2,2,2,  16,16,16,16,    16,16,16,16, 2,16],#M5
#[2,2,2,2,   2,2,2,2,  16,16,16,16,    16,16,16,16, 2,16],#M6

 [8,2,2,2,   2,2,2,2,  8,8,8,8,    8,8,8,8, 8,8],#M1
 #[8,2,2,2,   2,2,2,2,  8,8,8,8,    8,8,8,8, 4,4],#M2
 [8,2,2,2,   2,2,2,2,  8,8,8,8,    8,8,8,8, 2,8],#M3
 [2,2,2,2,   2,2,2,2,  8,8,8,8,    8,8,8,8, 2,8],#M4
]

exp_lstm_new2=[



 [8,4,4,4,   4,4,4,4,  8,4,4,4,    4,4,4,4, 4,4],#M7
 #[8,2,2,2,   4,4,4,4,  8,4,4,4,    4,4,4,4, 4,4],#M8
 #[8,4,4,4,    4,2,2,2,  8,4,4,4,    4,4,4,4, 4,4],#M9
 #[4,2,2,2,    4,4,4,4,  8,4,4,4,    4,4,4,4, 4,4],#M10
 #[8,4,4,4,    2,2,2,2,  8,4,4,4,    4,4,4,4, 4,4],#M11
 #[8,4,4,4,    4,2,2,2,  8,4,4,4,    4,4,4,4, 2,4],#M12
 [8,2,2,2,    2,2,2,2,  8,4,4,4,    4,4,4,4, 4,4],#M13
 [8,2,2,2,    2,2,2,2,  8,4,4,4,    4,4,4,4, 2,4],#M14
 [4,2,2,2,    4,2,2,2,  4,4,4,4,    4,4,4,4, 2,4],#M15
]
exp_gru_new_o=[
 [4, 2, 2, 2, 2, 4, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4],  # M15
 [16,16,16,16,16,  16,16,16,16,16, 16,16,16,16,16, 16,16,16,16,16,   16,16],
 [8,8,8,8,8, 8,8,8,8,8, 8,8,8,8,8,  8,8,8,8,8,  8,8],
 [4,4,4,4,4, 4,4,4,4,4, 4,4,4,4,4,  4,4,4,4,4, 4,4],
 [2,2,2,2,2, 2,2,2,2,2, 2,2,2,2,2,  2,2,2,2,2, 2,4],
 [16,2,2,2,2,   16,2,2,2,2,  16,16,16,16,16,    16,16,16,16,16, 16,16],#M1
 [16,2,2,2,2,   16,2,2,2,2,  16,16,16,16,16,    16,16,16,16,16, 4,4],#M2
 [16,2,2,2,2,   16,2,2,2,2,  16,4,4,4,4,        16,16,16,16,16,  16,16],#M3
 [16,2,2,2,2,   16,2,2,2,2,  16,16,16,16,16,   16,4,4,4,4,  16,16],#M4
 [16,2,2,2,2,   2,2,2,2,2,  16,16,16,16,16,    16,16,16,16,16, 2,16],#M5
 [2,2,2,2,2,   2,2,2,2,2,  16,16,16,16,16,    16,16,16,16,16, 2,16],#M6
 [8,4,4,4,4,   4,4,4,4,4,  8,4,4,4,4,    4,4,4,4,4, 4,4],#M7
 [8,2,2,2,2,   4,4,4,4,4,  8,4,4,4,4,    4,4,4,4,4, 4,4],#M8
 [8,4,4,4,4,    4,2,2,2,2,  8,4,4,4,4,    4,4,4,4,4, 4,4],#M9
 [4,2,2,2,2,    4,4,4,4,4,  8,4,4,4,4,    4,4,4,4,4, 4,4],#M10
 [8,4,4,4,4,    2,2,2,2,2,  8,4,4,4,4,    4,4,4,4,4, 4,4],#M11
 [8,4,4,4,4,    4,2,2,2,2,  8,4,4,4,4,    4,4,4,4,4, 2,4],#M12
 [8,2,2,2,2,    2,2,2,2,2,  8,4,4,4,4,    4,4,4,4,4, 4,4],#M13
 [8,2,2,2,2,    4,2,2,2,2,  8,4,4,4,4,    4,4,4,4,4, 2,4],#M14


]
def compute_mem_lstm(x,skip=0):
   if skip==0:
       return (x[0]*50600 + x[1]*2420000 + x[2]*2420000 +x[3]*2420000+ x[4]*1210000 +x[5]* 1210000+x[6]*1210000 +x[7]*1210000 +x[16]*2129600 + 4400*16)/8/1024/1024
   elif skip==1:
       return (x[0]*50600 + x[1]*2420000 + x[2]*2420000 +x[3]*2420000 + 3/4*(x[4]*1210000 +x[5]*1210000 +x[6]*1210000 +x[7]*1210000 )+16/4*(1210000*4)+x[16]*2129600 +4400 *16)/8/1024/1024
   elif skip == 2:
       return (x[0] * 50600 + x[1] * 2420000 + x[2] * 2420000 + x[3] * 2420000 + 3 / 4 * (
                   x[4] * 1210000 + x[5] * 1210000) + x[6] * 1210000 + x[7] * 1210000 + 16 / 4 * (1210000 * 2) + x[
                   16] * 2129600 + 4400 * 16) / 8 / 1024 / 1024

def compute_mem_gru(x,skip=0):
   if skip==0:
       return (x[0]*37950 + x[1]*1815000 + x[2]*1815000 +x[3]*1815000+x[4]*1815000+ x[5]*907500 +x[6]* 907500+x[7]*907500 +x[8]*907500+x[9]*907500 +x[20]*2129600 + 5500*16)/8/1024/1024
   elif skip==1:
       return (x[0]*37950 + x[1]*1815000 + x[2]*1815000 +x[3]*1815000 + x[4] * 1815000+ 2/3*(x[5]*907500 +x[6]*907500 +x[7]*907500 +x[8]*907500+x[9]*907500 )+16/3*(907500*5)+x[20]*2129600 +5500 *16)/8/1024/1024
   elif skip == 2:
       return (x[0] * 37950 + x[1] * 1815000 + x[2] * 1815000 + x[3] * 1815000+ x[4] * 1815000 + 2 / 3 * (
                   x[5] * 907500 + x[6] * 907500+ x[7] * 907500) + x[8] * 907500 + x[9] * 907500 + 16 / 3 * (907500 * 3) + x[
                   20] * 2129600 + 5500 * 16) / 8 / 1024 / 1024

def compute_mem_ligru(x,skip=0):
   if skip==0:
       return (x[0]*25300 + x[1]*1210000 + x[2]*1210000 +x[3]*1210000+x[4]*1210000+ x[5]*605000 +x[6]* 605000+x[7]*605000 +x[8]*605000+x[9]*605000 +x[20]*2129600 + 5500*16)/8/1024/1024
   elif skip==1:
       return (x[0]*25300 + x[1]*1210000 + x[2]*1210000 +x[3]*1210000 + x[4] * 1210000+ 2/3*(x[5]*605000 +x[6]*605000 +x[7]*605000 +x[8]*605000+x[9]*605000 )+16/3*(605000*5)+x[20]*2129600 +5500 *16)/8/1024/1024
   elif skip == 2:
       return (x[0] * 25300 + x[1] * 1210000 + x[2] * 1210000 + x[3] * 1210000+ x[4] * 1210000 + 2 / 3 * (
                   x[5] * 605000 + x[6] * 605000+ x[7] * 605000) + x[8] * 605000 + x[9] * 605000 + 16 / 3 * (605000 * 3) + x[
                   20] * 2129600 + 5500 * 16) / 8 / 1024 / 1024

def compute_sparsity_lstm(x,skip=0):
    if skip==0:
        return x[0]*0.00265+x[1]*0.1265+x[2]*0.1265+x[3]*0.1265+x[4]*0.1265+x[5]*0.1265+x[6]*0.1265+x[7]*0.1265+x[8]*0.1114+97*0.0002
    elif skip==1:
        return x[0] * 0.00265 + x[1] * 0.1265 + x[2] * 0.1265 + x[3] * 0.1265 + 3/4*(x[4] * 0.1265 + x[5] * 0.1265 + x[
            6] * 0.1265 + x[7] * 0.1265 ) + 1/4*0.1265*4 *97+ x[8] * 0.1114 + 97 * 0.0002
    elif skip==2:
        return x[0] * 0.00265 + x[1] * 0.1265 + x[2] * 0.1265 + x[3] * 0.1265 + 3/4*(x[4] * 0.1265 + x[5] * 0.1265) + x[
            6] * 0.1265 + x[7] * 0.1265  + 1/4*0.1265*2 *97+ x[8] * 0.1114 + 97 * 0.0002

def compute_sparsity_gru(x,skip=0):
    if skip==0:
        return x[0]*0.0021+x[1]*0.098+x[2]*0.098+x[3]*0.098+x[4]*0.098+\
               x[5]*0.098+x[6]*0.098+x[7]*0.098+x[8]*0.098+x[9]*0.098+\
               x[10]*0.115+97*0.0002
    elif skip==1:
        return x[0] * 0.0021 + x[1] * 0.098 + x[2] * 0.098 + x[3] * 0.098 + x[4] * 0.098 +\
               2/3*(x[5] * 0.098 + x[6] * 0.098 + x[7] * 0.098 + x[8] * 0.098+ x[9] * 0.098 ) +\
               1/3*0.098*5 *97+\
               x[10] * 0.115 + 97 * 0.0002
    elif skip==2:
        return x[0] * 0.0021 + x[1] * 0.098 + x[2] * 0.098 + x[3] * 0.098+ x[4] * 0.098 +\
               2/3*(x[5] * 0.098 + x[6] * 0.098+ x[7] * 0.098) + x[8] * 0.098 + x[9] * 0.098  +\
               1/3*0.098*3 *97+\
               x[10] * 0.115 + 97 * 0.0002

def compute_sparsity_ligru(x,skip=0):
    if skip==0:
        return x[0]*0.0019+x[1]*0.0927+x[2]*0.0927+x[3]*0.0927+x[4]*0.0927+x[5]*0.0927+x[6]*0.0927+x[7]*0.0927+x[8]*0.0927+x[9]*0.0927+x[10]*0.163+97*0.0002
    elif skip==1:
        return x[0] * 0.0019 + x[1] * 0.0927 + x[2] * 0.0927 + x[3] * 0.0927 + x[4] * 0.0927 + 1/2*(x[5] * 0.0927 + x[6] * 0.0927 + x[
            7] * 0.0927 + x[8] * 0.0927+ x[9] * 0.0927 ) + 1/2*0.0927*5 *97+ x[10] * 0.163 + 97 * 0.0002
    elif skip==2:
        return x[0] * 0.0019 + x[1] * 0.0927 + x[2] * 0.0927 + x[3] * 0.0927+ x[4] * 0.0927 + 1/2*(x[5] * 0.0927 + x[6] * 0.0927+ x[7] * 0.0927) + x[
            8] * 0.0927 + x[9] * 0.0927  + 1/2*0.0927*2 *97+ x[10] * 0.163 + 97 * 0.0002

#
delta=[
[0, 8,8,8,8,8,8,8,8],
[0, 16,16,16,16,16,16,16,16],
[0, 32,32,32,32,32,32,32,32]
]

delta2=[
[0, 0,0,0,0,0,0,0,0],
[0, 0,0,0,2,2,2,2,1]
]

#for i in exp_gru_new:
#    for j in delta2:
#        print(i)
#        print(j)
  #  wer, s = run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_GRU_fbank',
  #                                               i,
  #                                               delta=[0,0,0,0,0,0,0,0,0,0,0], n_layers=5, test=False, skip=0)
  #  wer, s = run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_GRU_fbank',
  #                                               i,
  #                                               delta=[0,0,0,0,0,0,0,0,0,0,0], n_layers=5, test=True, skip=0)
  #  print(compute_sparsity_gru(list(map(float, s.split(","))),1))
 #   print(compute_mem_gru(i))

   # wer, s = run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_GRU_fbank',
  #                                              i,
  #                                              delta=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], n_layers=5, test=False, skip=1)
  #  wer, s = run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_GRU_fbank',
  #                                              i,
  #                                              delta=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], n_layers=5, test=True, skip=1)
  #  print(compute_sparsity_gru(list(map(float, s.split(","))),1))
  #  print(compute_mem_gru(i,1))

  #  wer, s = run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_GRU_fbank',
  #                                              i,
  #                                           delta=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], n_layers=5, test=False, skip=1)
  #  wer, s = run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_GRU_fbank',
  #                                          i,
  #                                          delta=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], n_layers=5, test=True, skip=2)
  #  print(compute_sparsity_gru(list(map(float, s.split(","))),2))
  #  print(compute_mem_gru(i,2))

#wer, s = run_inference_for_optimization_liGRU('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_liGRU_fbank',
#                                                 [4, 2, 2,2,2, 2, 2, 2, 2, 2, 4, 2,2,2,2,2,2,2,2,2, 2, 4],
#                                                 delta=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], n_layers=5, test=False, skip=1)
#wer, s = run_inference_for_optimization_liGRU('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_liGRU_fbank',
#                                             [4, 2, 2, 2, 2, 2,2,2, 2, 2, 4, 2, 2, 2, 2,2,2, 2, 2, 2, 2, 4]
#                                             ,
#                                                 delta=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], n_layers=5, test=True, skip=1)
   # print(compute_sparsity_lstm(list(map(float, s.split(",")))))
   # print(compute_mem_lstm(i))

#        wer, s = run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_LSTM_fbank_c4',
#                                                     i,
#                                                     delta=j, n_layers=4, test=False, skip=1)
#        wer, s = run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_LSTM_fbank_c4',
#                                                     i,
#                                                     delta=j, n_layers=4, test=True, skip=1)
#        print(compute_sparsity_lstm(list(map(float, s.split(","))), skip=1))
#        print(compute_mem_lstm(i, 1))

#        wer, s = run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_LSTM_fbank_c4',
#                                                     i,
#                                                     delta=j, n_layers=4, test=False, skip=2)
#        wer, s = run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_LSTM_fbank_c4',
#                                                     i,
#                                                     delta=j, n_layers=4, test=True, skip=2)
#        print(compute_sparsity_lstm(list(map(float, s.split(","))), skip=2))
#        print(compute_mem_lstm(i, 2))


#for i in exp_lstm_new1:
    #for j in delta:
        #print(i)
        #print(j)
        #wer, s = run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_LSTM_fbank_c4',
         #                                            i,
         #                                            delta=j, n_layers=4, test=False, skip=0)
        #wer, s = run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_LSTM_fbank_c4',
        #                                             i,
        #                                             delta=j, n_layers=4, test=True, skip=0)
        #print(compute_sparsity_lstm(list(map(float, s.split(",")))))
        #print(compute_mem_lstm(i))

        #wer, s = run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_LSTM_fbank_c4',
        #                                             i,
        #                                             delta=j, n_layers=4, test=False, skip=1)
        #wer, s = run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_LSTM_fbank_c4',
        #                                             i,
        #                                             delta=j, n_layers=4, test=True, skip=1)
        #print(compute_sparsity_lstm(list(map(float, s.split(","))),skip=1))
        #print(compute_mem_lstm(i,1))

        #wer, s = run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_LSTM_fbank_c4',
        #                                             i,
        #                                             delta=j, n_layers=4, test=False, skip=2)
        #wer, s = run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_LSTM_fbank_c4',
        #                                          i,
        #                                             delta=j, n_layers=4, test=True, skip=2)
        #print(compute_sparsity_lstm(list(map(float, s.split(","))),skip=2))
        #print(compute_mem_lstm(i, 2))


#     wer, s = run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_LSTM_fbank_c4',
#                                                  [16, 2, 2, 2, 16, 2, 2, 2, 16, 8,8,8, 16, 8,8,8, 16, 16],
#                                                  delta=i, n_layers=4, test=False,skip=1)
#     wer, s = run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_LSTM_fbank_c4',
#                                                  [16, 2, 2, 2, 16, 2, 2, 2, 16, 8,8,8, 16, 8,8,8, 16, 16],
#                                                  delta=i, n_layers=4, test=True,skip=1)
#     print(compute_sparsity_lstm(list(map(float, s.split(","))), 1))

#     wer, s = run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_LSTM_fbank_c4',
#                                                  [16, 2, 2, 2, 16, 2, 2, 2, 16, 8,8,8, 16, 8,8,8, 16, 16],
#                                                  delta=i, n_layers=4, test=False,skip=2)
#     wer, s = run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_LSTM_fbank_c4',
#                                                  [16, 2, 2, 2, 16, 2, 2, 2, 16, 8,8,8, 16, 8,8,8, 16, 16],
#                                                  delta=i, n_layers=4, test=True,skip=2)
#     print(compute_sparsity_lstm(list(map(float, s.split(","))), 2))

# #    # i= [4,2,2,2,2,    4,2,2,2,2,  4,4,4,4,4,    4,4,4,4,4, 2,4],#M15
#      wer,s=run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_GRU_fbank',i,[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 5, False)
# #
# #
# # #    s='94.24748992919922,81.71247100830078,78.94451904296875,75.65583038330078,84.45455169677734,87.63636016845703,66.81818389892578,76.18181610107422,67.68108367919922'
# #
# # #    run_inference_for_optimization_liGRU('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_liGRU_fbank',i,[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 5, True)
# #  #   print(" case  "+str(i))
# #  #   print(compute_mem_ligru(i))
# #  #   print(compute_mem_ligru(i,1))
# #  #   print(compute_mem_ligru(i,2))
#      print(compute_sparsity_gru(list(map(float, s.split(",")))))
#      print(compute_sparsity_gru(list(map(float, s.split(","))),1))
#      print(compute_sparsity_gru(list(map(float, s.split(","))),2))
# #  #   run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_LSTM_fbank_c4',i,[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 4, True)
# #
# #     #print(compute_mem_lstm(i,2))
# #
# #
#
#wer, s = run_inference_for_optimization('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_SRU_fbank_dnn_6', [ 8  ,2   ,   4 , 4  ,  8 ,16 ,   4 ,16   , 2 ,8  ,2       ,16, 8 , 4        ,2  ,8],
#                                        delta=[0, 0, 0, 0, 0, 0, 0, 0], n_layers=4, test=False, opt_index=4)
#wer_t, s = run_inference_for_optimization('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_SRU_fbank_dnn_6', [ 8  ,2   ,   4 , 4  ,  8 ,16 ,   4 ,16   , 2 ,8  ,2       ,16, 8 , 4        ,2  ,8],
#                                          delta=[0, 0, 0, 0, 0, 0, 0, 0], n_layers=4, test=True)

#wer, s = run_inference_for_optimization('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_SRU_fbank_dnn_6', [ 8 , 4 ,    4 , 2  ,     8, 16,     8, 16   ,     2,    8 , 2    ,   16   ,8 , 4 ,      2 , 4 ],
#                                        delta=[0, 0, 0, 0, 0, 0, 0, 0], n_layers=4, test=False, opt_index=4)
#wer_t, s = run_inference_for_optimization('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_SRU_fbank_dnn_6', [8 , 4 ,    4 , 2  ,     8, 16,     8, 16   ,     2,    8 , 2    ,   16   ,8 , 4 ,      2 , 4 ],
#                                          delta=[0, 0, 0, 0, 0, 0, 0, 0], n_layers=4, test=True)
#wer,s=run_inference_for_optimization('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_SRU_fbank_dnn_l6', [ 8 , 2  ,2,  2,  4  ,4,    8 ,16,  8 , 4  ,4, 16    ,2 , 2  ,8  ,2  ,8 ,  16 ,16 , 8   ,4 ,16 , 2 , 8],delta=[0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0], n_layers=6, test=False,opt_index=4)
#wer_t,s=run_inference_for_optimization('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_SRU_fbank_dnn_l6',  [ 8 , 2  ,2,  2,  4  ,4,  8 ,16,  8 , 4  ,4, 16  ,2 , 2  ,8  ,2  ,8 ,16 ,16 , 8   ,4 ,16 , 2 , 8],delta=[0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0], n_layers=6, test=True)

#wer,s=run_inference_for_optimization('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_SRU_fbank_dnn_l6',  [ 4 , 4  ,2 , 2,  4 , 4 , 8 ,16 , 8 , 4  ,2  ,8 , 2  ,2 ,16,  2  ,8 , 4, 16 , 4 , 16 ,16  ,2,  4],delta=[0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0], n_layers=6, test=False,opt_index=4)
#wer_t,s=run_inference_for_optimization('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_SRU_fbank_dnn_l6',  [ 4 , 4  ,2 , 2,  4 , 4 , 8 ,16 , 8 , 4  ,2  ,8 , 2  ,2 ,16,  2  ,8 , 4, 16 , 4 , 16 ,16  ,2,  4],delta=[0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0], n_layers=6, test=True)

#wer_t,s=run_inference_for_optimization('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_SRU_fbank_dnn_l6',  [8  ,4,  2  ,2  ,4 , 2          ,8 ,16  ,8  ,4  ,8, 16         ,2  ,2  ,8  ,2,  2       ,16 ,16,  8  ,4, 16   ,     2 , 4],delta=[0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0], n_layers=6, test=True)

#[ 8  2      4  4    8 16    4 16    2 8  2       16 8  4        2  8]  #15.4 16.9
#[ 4  4     4  4    8 16    2  8    2 16  2       4  4  16          2  4] #15.1  17.8

#[ 8  2    4  4       8 16     4 16        2    8  2       16   8   4      2  8]  #15.4 16.9
#[ 4  4     4  4       8 16     2  8        2   16  2       4   4  16       2  4] #15.1 17.8
#[ 8  4     4  2       8 16     8 16        2    8  2       16   8  4       2  4] #15.7 17
#[ 4  4  2  2  4  4          8 16  8  4  2  8         2  2 16  2  8       4 16  4  16 16        2  4]
#[ 8  2  2  2  4  4          8 16  8  4  4 16         2  2  8  2  8       16 16  8   4 16       2  8]
#[ 8  4  2  2  4  2          8 16  8  4  8 16         2  2  8  2  2       16 16  8  4 16        2  4]
