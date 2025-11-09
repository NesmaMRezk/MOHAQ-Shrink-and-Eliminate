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
def set_bits_conf(config,w0,w1,w2,w3,iw0,iw1,iw2,iw3,p0,p1,p2,p3,ip1,ip2,ip3,fc,fci):
    config.set("architecture1", "weights_bits", str(w0) + "," + str(w1) + "," + str(w2) + "," + str(w3))
    config.set("architecture1", "inputs_bits", "16," + str(ip1) + "," + str(ip2) + "," + str(ip3))
    config.set("architecture1", "projection_bits", "16," + str(p1) + "," + str(p2) + "," + str(p3))
    config.set("architecture1", "weights_activation_bits", str(iw0) + "," + str(iw1) + "," + str(iw2) + "," + str(iw3))
    config.set("architecture1", "output_bits", str(fci))
    config.set("architecture2", "number_of_bits", str(fc))
#Nesma This function changes in the configuration files to enable the use of the optimization dataset either for optimization or collecting statistics

def set_val_conf(kaldi_path,conf_main,config,outfolder):
    config.set("architecture1", "use_statistics", "False")
    config.set("data_chunk", "fea",
               "fea_name=fbank \n fea_lst="+kaldi_path+"/egs/timit/s5/data/dev_o/feats.scp	\n fea_opts=apply-cmvn --utt2spk=ark:"+kaldi_path+"/egs/timit/s5/data/dev_o/utt2spk ark:"+kaldi_path+"/egs/timit/s5/fbank/cmvn_dev.ark ark:- ark:- | add-deltas --delta-order=0 ark:- ark:- |  \n cw_left=0 \n cw_right=0")
    config.set("data_chunk", "lab",
               " lab_name=lab_cd \n lab_folder="+kaldi_path+"/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali_dev \n lab_opts=ali-to-pdf \n lab_count_file=auto \n lab_data_folder="+kaldi_path+"/egs/timit/s5/data/dev_o/ \n lab_graph="+kaldi_path+"/egs/timit/s5/exp/tri3/graph \n \n \n lab_name=lab_mono \n lab_folder="+kaldi_path+"/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali_dev \n lab_opts=ali-to-phones --per-frame=true \n lab_count_file=none \n lab_data_folder="+kaldi_path+"/egs/timit/s5/data/dev_o/ \n lab_graph="+kaldi_path+"/egs/timit/s5/exp/tri3/graph")
    config.set("exp","out_info",outfolder+"/exp_files/forward_TIMIT_opt_ep23_ck0.info")

    conf_main.set("data_use", "forward_with", "TIMIT_opt")

def set_val_conf_full(kaldi_path,conf_main,config,outfolder):
    config.set("architecture1", "use_statistics", "False")
    config.set("data_chunk", "fea",
               "fea_name=fbank \n fea_lst="+kaldi_path+"/egs/timit/s5/data/train_o2/feats.scp	\n fea_opts=apply-cmvn --utt2spk=ark:"+kaldi_path+"/egs/timit/s5/data/train_o2/utt2spk ark:"+kaldi_path+"/egs/timit/s5/fbank/cmvn_train.ark ark:- ark:- | add-deltas --delta-order=0 ark:- ark:- |  \n cw_left=0 \n cw_right=0")
    config.set("data_chunk", "lab",
               " lab_name=lab_cd \n lab_folder="+kaldi_path+"/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali \n lab_opts=ali-to-pdf \n lab_count_file=auto \n lab_data_folder="+kaldi_path+"/egs/timit/s5/data/train_o2/ \n lab_graph="+kaldi_path+"/egs/timit/s5/exp/tri3/graph \n \n \n lab_name=lab_mono \n lab_folder="+kaldi_path+"/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali \n lab_opts=ali-to-phones --per-frame=true \n lab_count_file=none \n lab_data_folder="+kaldi_path+"/egs/timit/s5/data/train_o2/ \n lab_graph="+kaldi_path+"/egs/timit/s5/exp/tri3/graph")
    config.set("exp","out_info",outfolder+"/exp_files/forward_TIMIT_tr_op_ep23_ck0.info")

    conf_main.set("data_use", "forward_with", "TIMIT_tr_op")
#Nesma This function is changes in the configuration file to use the test set and use the collected statistics instead of computing the maximum values in the activations vectors
def set_test_conf(kaldi_path,conf_main,config,outfolder):
    config.set("architecture1", "use_statistics", "True")

    config.set("data_chunk", "fea",
               "fea_name=fbank \n fea_lst="+kaldi_path+"/egs/timit/s5/data/test/feats.scp	\n fea_opts=apply-cmvn --utt2spk=ark:"+kaldi_path+"/egs/timit/s5/data/test/utt2spk ark:"+kaldi_path+"/egs/timit/s5/fbank/cmvn_test.ark ark:- ark:- | add-deltas --delta-order=0 ark:- ark:- |  \n cw_left=0 \n cw_right=0")
    config.set("data_chunk", "lab",
               " lab_name=lab_cd \n lab_folder="+kaldi_path+"/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali_test \n lab_opts=ali-to-pdf \n lab_count_file=auto \n lab_data_folder="+kaldi_path+"/egs/timit/s5/data/test \n lab_graph="+kaldi_path+"/egs/timit/s5/exp/tri3/graph \n \n \n lab_name=lab_mono \n lab_folder="+kaldi_path+"/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali_test \n lab_opts=ali-to-phones --per-frame=true \n lab_count_file=none \n lab_data_folder="+kaldi_path+"/egs/timit/s5/data/test \n lab_graph="+kaldi_path+"/egs/timit/s5/exp/tri3/graph")
    config.set("exp","out_info",outfolder+"/exp_files/forward_TIMIT_test_ep23_ck0.info")
    #TODO create validate for triaining in the conf file
    conf_main.set("data_use","forward_with","TIMIT_test")


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
def run_inference_for_optimization(kaldi_path,out_folder,w0,w1,w2,w3,iw0,iw1,iw2,iw3,p1,p2,p3,ip1,ip2,ip3,fc,fci,test,full=False):


    layers_bits = [w0, w1, w2, w3]
    proj_bits = [16, p1, p2, p3]
    layer_bits=[4,4]
    proj_bits=[]
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
    set_bits_conf(config, w0,w1,w2,w3,iw0, iw1, iw2, iw3,16,p1,p2,p3,ip1, ip2, ip3,fc, fci)

    if test==True:
        set_test_conf(kaldi_path,config_main,config,out_folder)
    elif full == False:
        set_val_conf(kaldi_path,config_main, config,out_folder)
    else:
        set_val_conf_full(kaldi_path, config_main, config, out_folder)
    with open(out_folder + '/quantize_TIMIT.cfg', 'w') as f:
        config.write(f)
    with open(cfg_file, 'w') as h:
        config_main.write(h)

    # quantize the sru model
    quantize(out_folder, 1, 'LSTM', layers_bits,proj_bits)
    # quantize the FC layer
    quantize(out_folder, 2, 'MLP', [fc])


    return float(run_inference(cfg_file))

#out_folder = "exp/TIMIT_SRU_fbank_z"
#quantize(out_folder, 1, 'SRU', [4,4,4,4],[16,16,16,16])
#quantize(out_folder, 2, 'MLP', [8])
#run_inference("cfg/TIMIT_baselines/TIMIT_SRU_fbank.cfg")
