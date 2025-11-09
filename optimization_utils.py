#Nesma
#Utilazation functions required during optimization
import configparser
import os
import sys
from math import sqrt

#
from pandas.tests.io.excel.test_xlsxwriter import xlsxwriter

from run_exp import retrain, retrain_lstm, retrain_LiGRU
from run_exp_test import run_inference_for_optimization, run_inference_for_optimization_lstm_delta, \
    run_inference_for_optimization_liGRU
from utils import run_shell

cfg_file = "cfg/TIMIT_baselines/TIMIT_liGRU_fbank.cfg"
if not (os.path.exists(cfg_file)):
    sys.stderr.write("ERROR: The config file %s does not exist!\n" % (cfg_file))
    sys.exit(0)
else:

    config_main = configparser.ConfigParser()
    config_main.read(cfg_file)


beacon_sheet_counter=0
Kaldi_path= os.environ['KALDI_ROOT']
list_base_6=[16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16]
list_base=[16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16]
list_2_base=[2,2,2,2,16,16,16,16,2,2,2,16,16,16,16,16]
list_4_base=[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]
list_8_base=[8,8,8,8,4,4,4,4,8,8,8,4,4,4,16,16]
list_15_base=[16,2,2,2,16,16,16,16,4,4,4,16,16,16,16,16]
temp_1=[4,4,2,8,4,16,16,8,2,8,16,16,4,16,16,8]
temp_3=[4,16,16,8,16,16,2,16,2,8,8,8,16,2,4,8]
temp_4=[4,4,2,16,16,4,16,4,16,2,8,4,16,8,8,8]
temp_5=[8,4,2,8,16,2,2,16,4,2,16,4,16,8,16,16]
temp_6=[4,16,16,8,16,16,16,8,2,4,8,4,16,2,4,8]
temp_134=[2,2,8,8,8,4,8,16,8,8,2,8,4,16,2,4]
temp_111=[2,16,2,8,4,16,4,8,2,4,4,8,2,8,16,16]
temp_109=[16,8,8,8,4,8,16,16,4,2,16,8,2,16,2,4]
temp_106=[16,8,2,8,8,16,4,4,4,8,2,8,16,16,2,4]
temp_108=[4,16,8,16,16,16,16,8,16,4,4,8,2,8,16,16]
temp_104=[8,4,4,4,8,4,4,4,16,8,2,16,16,16,4,8]
temp_170=[2,8,4,2,4,16,8,8,16,8,4,8,2,8,16,16]
temp_141=[4,8,2,2,8,4,8,4,2,8,2,16,4,2,2,8]

#beacons=[list_2_base,list_4_base,list_8_base,list_15_base,list_04]
#beacons=[list_base,list_2_base,list_4_base,list_8_base,list_15_base,temp_1,temp_3,temp_4,temp_5,temp_6,temp_134,temp_111,temp_109,temp_106,temp_108,temp_104,temp_170,temp_141]
#beacons_folders=["exp/TIMIT_SRU_fbank_base","exp/TIMIT_SRU_fbank_y/retrain_2","exp/TIMIT_SRU_fbank_y/retrain_4", "exp/TIMIT_SRU_fbank_y/retrain_8_4","exp/TIMIT_SRU_fbank_y/retrain_15","exp/TIMIT_SRU_fbank_y/retrain_0y101","exp/TIMIT_SRU_fbank_y/retrain_0y103","exp/TIMIT_SRU_fbank_y/retrain_0y104","exp/TIMIT_SRU_fbank_y/retrain_0y105","exp/TIMIT_SRU_fbank_y/retrain_0y106","exp/TIMIT_SRU_fbank_y/retrain_0zz134","exp/TIMIT_SRU_fbank_y/retrain_0zz111","exp/TIMIT_SRU_fbank_y/retrain_0zz109","exp/TIMIT_SRU_fbank_y/retrain_0zz106","exp/TIMIT_SRU_fbank_y/retrain_0zz108","exp/TIMIT_SRU_fbank_y/retrain_0zz104","exp/TIMIT_SRU_fbank_y/retrain_0zz170","exp/TIMIT_SRU_fbank_y/retrain_0zz141"]
#beacons_folders=["exp/TIMIT_SRU_fbank_y/retrain_2","exp/TIMIT_SRU_fbank_y/retrain_4", "exp/TIMIT_SRU_fbank_y/retrain_8_4","exp/TIMIT_SRU_fbank_y/retrain_15","exp/TIMIT_SRU_fbank_y/retrain_046"]
#list_86=[16,2,8,8,4,2,16,4,16,4,8,2,2,16,2,4]
list_100=[4,2,2,4,4,2,2,16,4,2,4,4,8,2,2,8]
list_326=[8,8,4,2,8,4,2,2,8,2,2,2,4,16,2,4]
list_2222=[2,2,2,2,16,16,16,16,2,2,2,16,16,16,16]
list_2=[4,2,4,4,16,2,2,2,4,2,4,2,4,8,2,16]

beacon_164=[2,2,2,2,2,2,  16,4,2,4,8,8,  16,2,2,2,16,  4,2,16,8,4,  2,4]
beacon_227=[8,8,4,2,2,2,  16,2,8,2,16,2, 2,2,2,8,2,    16,16,4,2,2, 2,8]
beacon_228=[8,2,2,4,2,2,  16,4,4,8,8,2,  2,8,4,8,8,    2,4,4,2,4,   2,8]
beacon_338=[4,8,2,4,2,2,  16,4,4,4,4,2,  4,4,2,4,2,    2,8,4,2,4,   2,8]
beacon_399=[16,2,2,4,2,2, 8,4,8,2,4,4,   2,8,2,4,16,   2,16,4,2,4,  2,8]
beacon_465=[4,2,2,2,2,2,  8,4,4,2,2,4,   8,4,8,4,8,    2,4,4,4,4,   2,4]
beacon_600=[8,2,2,2,2,2,  4,4,4,4,4,4,   2,2,2,2,2,    4,4,4,4,4,  2,16]

beacon_156=[16,2,2,2,  16,2,2,2,  2,2,2,  2,2,4,  2,8]
beacon_152=[4,2,4,2,   16,4,8,2,  4,2,2,  2,2,16, 2,8]
beacon_199=[8,2,2,2,   16,8,16,2, 8,8,2,  2,2,8,  2,16]
beacon_192=[2,2,2,2,   16,8,16,2, 8,8,2,  2,2,8,  2,16]


beaconG_156=[16,2,2,2,2,  2,2,2,2,2,  16,2,2,2,2,  2,2,4,4,4,  2,8]
beaconG_152=[4,2,4,2,2,   4,2,2,2,2,  8,2,2,4,2,   2,2,2,2,8, 2,4]
beaconG_1522=[4,2,2,2,2,   2,2,2,2,2,  8,2,2,2,2,   2,2,2,2,2, 2,4]
beaconG_199=[8,2,2,2,2,    8,8,2,2,2, 16,8,16,2,2,  2,2,8,2,4,  2,8]
beaconG_192=[2,2,4,2,2,  8,8,2,2,2,  16,8,16,2,2,  2,2,4,8,4,  2,8]
beaconsG_folders=['exp/TIMIT_liGRU_fbank','exp/retrainingG/retrain_rrx152','exp/retrainingG/retrain_rrx156','exp/retrainingG/retrain_rrx192']
beaconsG=[list_base,beacon_152,beacon_192,beacon_156]

beacons=[list_base,beacon_152,beacon_199,beacon_192,beacon_156]
beacons_folders=['exp/TIMIT_SRU_fbank_dnn_6','exp/retraining/retrain_rrx152','exp/retraining/retrain_rrx199','exp/retraining/retrain_rrx192','exp/retraining/retrain_rrx156']
#beacons_6=[list_base_6,beacon_164,beacon_227,beacon_228,beacon_338,beacon_399,beacon_465,beacon_600,beacon_465,beacon_600]
#beacons_6=[list_base_6,beacon_399,beacon_465,beacon_399,beacon_600]
#beacons_folders_6=['exp/TIMIT_SRU_fbank_dnn_l6','exp/retraining_6l/retrain_rrx164','exp/retraining_6l/retrain_rrx227','exp/retraining_6l/retrain_rrx228','exp/retraining_6l/retrain_rrx338','exp/retraining_6l/retrain_rrx399','exp/retraining_6l/retrain_rrx465','exp/retraining_6l/retrain_rrx600','exp/retraining_6l/retrain_rrx6000','exp/retraining_6l/retrain_rrx60000']
#beacons_folders_6=['exp/TIMIT_SRU_fbank_dnn_l6','exp/retraining_6l/retrain_rrx3990','exp/retraining_6l/retrain_rrx4650','exp/retraining_6l/retrain_rrx3990000','exp/retraining_6l/retrain_rrx6000000','exp/retraining_6l/retrain_rrx199','exp/retraining_6l/retrain_rrx152']

beacon_198_6=[8,2,2,2,2,2,  16,2,2,2,4,2, 8,8,2,8,2, 2,2,2,2,4,   2,16]#6  loser high loss >199,159,152

beacon_159_6=[4,2,4,2,4,2,  16,2,2,2,2,2,   4,2,2,2,2,  2,2,2,2,4,   2,16]#5
beacon_152_6=[4,2,4,2,4,2,  16,4,8,2,8,2,   4,2,2,2,2,  2,2,2,2,16,  2,16]#1
beacon_199_6=[8,2,2,2,2,2,  16,8,16,2,16,2, 8,8,2,8,2,  2,2,8,2,8,   2,16]#2

beacon_194_6=[8,2,2,2,2,2,  16,8,16,2,16,2, 8,8,2,8,2, 2,2,8,2,8,   2,4]#8 loss slightly > 152,199,159
beacon_156_6=[16,2,4,2,2,2,  16,2,4,2,4,2,  4,2,2,2,2,  2,2,2,2,4,  2,16]#9 used when first layer 16,16
beacon_158_6=[8,2,4,2,4,2,  16,4,8,2,8,2,  4,2,2,2,2,  2,2,2,2,16,  2,16]#4
beacon_192_6=[2,2,2,2,2,2,  16,8,16,2,16,2, 8,8,2,8,2, 2,2,8,2,8,   2,16]#3 used when first layer w =2
beacon_100_6=[4,4,8,16,4,4, 4,4,8,2,8,8,   4,4,4,16,8, 16,8,4,4,16,  16,4] # no need
beacon_102_6=[4,2,8,2,2,4, 4,4,8,8,8,8,   4,2,4,2,8, 16,8,8,4,16,  16,4] #7 no need, it was a bug

beacons_6=[list_base_6,beacon_199_6,beacon_192_6,beacon_156_6,beacon_152_6,beacon_158_6,beacon_159_6,beacon_198_6,beacon_102_6,beacon_194_6,beacon_156_6]
beacons_folders_6=['exp/TIMIT_SRU_fbank_dnn_l6','exp/retraining_6l/retrain_rrx199','exp/retraining_6l/retrain_rrx192','exp/retraining_6l/retrain_rrx156','exp/retraining_6l/retrain_rrx152','exp/retraining_6l/retrain_rrx159','exp/retraining_6l/retrain_rrx198','exp/retraining_6l/retrain_rrx102','exp/retraining_6l/retrain_rrx194','exp/retraining_6l/retrain_rrx156']

list_base_3=[16,16,16,16,16,16,16,16,16,16,16,16]

beacons_3=[list_base_3,[4,2,2,4,4,4,2,2,4,4,2,4],[8,2,2,16,8,2,8,2,2,8,2,16],[2,2,2,16,8,2,8,2,2,8,2,16]]
beacons_folders_3=['exp/TIMIT_SRU_fbank_dnn_l3','exp/retraining_3l/retrain_rrx152','exp/retraining_3l/retrain_rrx199000','exp/retraining_3l/retrain_rrx192']
def encode(x):
    if x == 1:
        return 2
    elif x== 2:
        return 4
    elif x == 3:
        return 8
    elif x==4:
        return 16
    else:
        return -1

def decode(x):
    if x== 1:
        return 1
    if x==2:
        return 2
    if x==4:
        return 3
    if x==8:
        return 4
    if x==16:
        return 5
    return 0
def compute_distance(l1,l2):
    distance=0
    for i in range(8):
        if i>3:
            i+=4
        if i==11:
            i=14

       # if(l1[i]!=l2[i]):
       #     distance+=1
       # if i==0:
       #     if decode(l1[0])!=decode(l2[0]):
       #         distance+=2
        distance+=abs(decode(l1[i])-decode(l2[i]))


    return (distance)

def compute_distance_6(l1,l2):
    distance=0

    for i in range(12):
        if i>5:
            i+=6
        if i==17:
            i=22

       # if(l1[i]!=l2[i]):
       #     distance+=1
       # if i==0:
       #     if decode(l1[0])!=decode(l2[0]):
       #         distance+=2

        distance+=abs(decode(l1[i])-decode(l2[i]))


    return (distance)

def compute_distance_act_6(l1,l2):
    distance=0

    for i in range(12):
        if i<=5:
            i+=6
        elif i<11:
            i+=11
        elif i==11:
            i=23

       # if(l1[i]!=l2[i]):
       #     distance+=1
       # if i==0:
       #     if decode(l1[0])!=decode(l2[0]):
       #         distance+=2

        distance+=abs(decode(l1[i])-decode(l2[i]))


    return (distance)

def compute_distance_3(l1,l2):
    distance=0

    for i in range(6):
        if i>2:
            i+=3
        if i==8:
            i=10

       # if(l1[i]!=l2[i]):
       #     distance+=1
       # if i==0:
       #     if decode(l1[0])!=decode(l2[0]):
       #         distance+=2

        distance+=abs(decode(l1[i])-decode(l2[i]))


    return (distance)
def convert_to_beacon_G(wer,x,i):
    #steps before
    #core.py

    excel_b = xlsxwriter.Workbook('retrained-beacons_'+str(i)+'.xlsx')
    beacons_sheet = excel_b.add_worksheet()
    folder="exp/retrainingG/retrain_rrx"+str(i)
    old_folder="exp/TIMIT_liGRU_fbank"

 #   beacon_sheet_counter+=1
    beacons_sheet.write("A1", ','.join(str(z) for z in x))
    excel_b.close()
    if  True:#not (os.path.exists(folder)):
        #os.popen('mkdir '+folder)
        #os.popen('mkdir ' + folder+'/exp_files')
        #os.popen('cp  exp/retrainingG/pretrained/* ' + folder + "/exp_files/")
        #os.popen('cp  exp/TIMIT_liGRU_fbank/quantize_TIMIT.cfg  ' + folder + "/quantize_TIMIT.cfg")
        #x[len(x)-1]=16
        if wer<19:
            retrain_LiGRU(Kaldi_path, folder, old_folder, x,n_layers=5,n_epochs=11)
        else:
            retrain_LiGRU(Kaldi_path, folder,old_folder, x,n_layers=5, n_epochs=11)
        #os.popen('cp /usr/local/home/nesrez/conda/envs/env/lib/python3.7/site-packages/sru-2.5.1-py3.7.egg/sru/modules_i.py /usr/local/home/nesrez/conda/envs/env/lib/python3.7/site-packages/sru-2.5.1-py3.7.egg/sru/modules.py')


        config = configparser.ConfigParser()
        config.read(folder + '/quantize_TIMIT.cfg')
        config.set("exp", "out_folder", folder)
        config.set("architecture1","arch_pretrain_file",folder+"/exp_files/quantize_TIMIT_architecture1.pth.tar")
        config.set("architecture2","arch_pretrain_file",folder+"/exp_files/quantize_TIMIT_architecture2.pth.tar")
        config.set("architecture3","arch_pretrain_file",folder+"/exp_files/final_architecture3.pth.tar")
        config.set("architecture1","conf_file","/usr/local/home/nesrez/pytorch-kaldi/"+folder+"/quantize_TIMIT.cfg")
        with open(folder + '/quantize_TIMIT.cfg', 'w') as f:
            config.write(f)

    beaconsG.append(x)
    beaconsG_folders.append(folder)


def convert_to_beacon_v2(wer,x,i):
    #steps before
    #core.py

    excel_b = xlsxwriter.Workbook('retrained-beacons_'+str(i)+'.xlsx')
    beacons_sheet = excel_b.add_worksheet()
    folder="exp/retraining_6l/retrain_rrx"+str(i)
    old_folder="exp/TIMIT_SRU_fbank_dnn_l6"

 #   beacon_sheet_counter+=1
    beacons_sheet.write("A1", ','.join(str(z) for z in x))
    excel_b.close()
    if  not (os.path.exists(folder)):
        os.popen('mkdir '+folder)
        os.popen('mkdir ' + folder+'/exp_files')
        os.popen('cp  exp/retraining_6l/pretrained/* ' + folder + "/exp_files/")
        os.popen('cp  exp/TIMIT_SRU_fbank_dnn_l6/quantize_TIMIT.cfg  ' + folder + "/quantize_TIMIT.cfg")
        #x[len(x)-1]=16
        if wer<19:
            retrain(Kaldi_path, folder, old_folder, x,n_layers=6,n_epochs=20)
        else:
            retrain(Kaldi_path, folder,old_folder, x,n_layers=6, n_epochs=20)
        #os.popen('cp /usr/local/home/nesrez/conda/envs/env/lib/python3.7/site-packages/sru-2.5.1-py3.7.egg/sru/modules_i.py /usr/local/home/nesrez/conda/envs/env/lib/python3.7/site-packages/sru-2.5.1-py3.7.egg/sru/modules.py')


        config = configparser.ConfigParser()
        config.read(folder + '/quantize_TIMIT.cfg')
        config.set("exp", "out_folder", folder)
        config.set("architecture1","arch_pretrain_file",folder+"/exp_files/quantize_TIMIT_architecture1.pth.tar")
        config.set("architecture2","arch_pretrain_file",folder+"/exp_files/quantize_TIMIT_architecture2.pth.tar")
        config.set("architecture3","arch_pretrain_file",folder+"/exp_files/final_architecture3.pth.tar")
        config.set("architecture1","conf_file","/usr/local/home/nesrez/pytorch-kaldi/"+folder+"/quantize_TIMIT.cfg")
        with open(folder + '/quantize_TIMIT.cfg', 'w') as f:
            config.write(f)

    beacons_6.append(x)
    beacons_folders_6.append(folder)
def convert_to_beacon_v3(wer,x,i):
    #steps before
    #core.py

    excel_b = xlsxwriter.Workbook('retrained-beacons_3l_'+str(i)+'.xlsx')
    beacons_sheet = excel_b.add_worksheet()
    folder="exp/retraining_3l/retrain_rrx"+str(i)
    old_folder="exp/TIMIT_SRU_fbank_dnn_l3"

 #   beacon_sheet_counter+=1
    beacons_sheet.write("A1", ','.join(str(z) for z in x))
    excel_b.close()
    if  not (os.path.exists(folder)):
        os.popen('mkdir '+folder)
        os.popen('mkdir ' + folder+'/exp_files')
        os.popen('cp  exp/retraining_3l/pretrained/* ' + folder + "/exp_files/")
        os.popen('cp  exp/TIMIT_SRU_fbank_dnn_l3/quantize_TIMIT.cfg  ' + folder + "/quantize_TIMIT.cfg")
        x[len(x)-1]=16
        if wer<19:
            retrain(Kaldi_path, folder, old_folder, x,n_layers=3,n_epochs=20)
        else:
            retrain(Kaldi_path, folder,old_folder, x,n_layers=3, n_epochs=20)
        #os.popen('cp /usr/local/home/nesrez/conda/envs/env/lib/python3.7/site-packages/sru-2.5.1-py3.7.egg/sru/modules_i.py /usr/local/home/nesrez/conda/envs/env/lib/python3.7/site-packages/sru-2.5.1-py3.7.egg/sru/modules.py')


        config = configparser.ConfigParser()
        config.read(folder + '/quantize_TIMIT.cfg')
        config.set("exp", "out_folder", folder)
        config.set("architecture1","arch_pretrain_file",folder+"/exp_files/quantize_TIMIT_architecture1.pth.tar")
        config.set("architecture2","arch_pretrain_file",folder+"/exp_files/quantize_TIMIT_architecture2.pth.tar")
        config.set("architecture3","arch_pretrain_file",folder+"/exp_files/final_architecture3.pth.tar")
        config.set("architecture1","conf_file","/usr/local/home/nesrez/pytorch-kaldi/"+folder+"/quantize_TIMIT.cfg")
        with open(folder + '/quantize_TIMIT.cfg', 'w') as f:
            config.write(f)

    beacons_3.append(x)
    beacons_folders_3.append(folder)

def convert_to_beacon(wer,x,i):
    #steps before
    #core.py

    excel_b = xlsxwriter.Workbook('retrained-beacons_'+str(i)+'.xlsx')
    beacons_sheet = excel_b.add_worksheet()
    folder="exp/retraining/retrain_rrx"+str(i)
    old_folder="exp/TIMIT_SRU_fbank_dnn_6"

 #   beacon_sheet_counter+=1
    beacons_sheet.write("A1", ','.join(str(z) for z in x))
    excel_b.close()
    if not (os.path.exists(folder)):
        os.popen('mkdir '+folder)
        os.popen('mkdir ' + folder+'/exp_files')
        os.popen('cp  exp/TIMIT_SRU_fbank_dnn_6/quantize_TIMIT.cfg  ' + folder + "/quantize_TIMIT.cfg")
        os.popen('cp  exp/TIMIT_SRU_fbank_dnn_6/pretrained/* ' + folder + "/exp_files/")

        x[len(x) - 1] = 16
        if wer<20:
            x[4]=4
            x[5]=4
            x[6] = 4
            x[7] = 4

            x[11] = 4
            x[12] = 4
            x[13] = 4
            retrain(Kaldi_path, folder, old_folder, x,n_epochs=20,n_layers=4)
        else:
#
            retrain(Kaldi_path, folder,old_folder, x, n_epochs=20,n_layers=4)
        #os.popen('cp /usr/local/home/nesrez/conda/envs/env/lib/python3.7/site-packages/sru-2.5.1-py3.7.egg/sru/modules_i.py /usr/local/home/nesrez/conda/envs/env/lib/python3.7/site-packages/sru-2.5.1-py3.7.egg/sru/modules.py')

        config = configparser.ConfigParser()
        config.read(folder + '/quantize_TIMIT.cfg')
        config.set("exp", "out_folder", folder)
        config.set("architecture1","arch_pretrain_file",folder+"/exp_files/quantize_TIMIT_architecture1.pth.tar")
        config.set("architecture2","arch_pretrain_file",folder+"/exp_files/quantize_TIMIT_architecture2.pth.tar")
        config.set("architecture3","arch_pretrain_file","exp/TIMIT_SRU_fbank_dnn_6/exp_files/final_architecture3.pth.tar")
        config.set("architecture1","conf_file","/usr/local/home/nesrez/pytorch-kaldi/"+folder+"/quantize_TIMIT.cfg")
        with open(folder + '/quantize_TIMIT.cfg', 'w') as f:
            config.write(f)

    beacons.append(x)
    beacons_folders.append(folder)
def convert_to_beacon_LSTM(wer,x,i):
    #steps before
    #core.py

    excel_b = xlsxwriter.Workbook('retrained-beacons_'+str(i)+'.xlsx')
    beacons_sheet = excel_b.add_worksheet()
    folder="exp/retraining_lstm/retrain_rrx"+str(i)
    old_folder="exp/TIMIT_LSTM_fbank_c4"

 #   beacon_sheet_counter+=1
    beacons_sheet.write("A1", ','.join(str(z) for z in x))
    excel_b.close()
    if  not (os.path.exists(folder)):

        os.popen('mkdir '+folder)
        os.popen('mkdir ' + folder+'/exp_files')
           # os.popen('cp -r exp/TIMIT_SRU_fbank_dnn_6  '+folder+"/")
        os.popen('cp  exp/TIMIT_LSTM_fbank_c4/exp_files/final_architecture1.pth.tar ' + folder + "/exp_files/final_architecture1.pth.tar")
        os.popen('cp  exp/TIMIT_LSTM_fbank_c4/exp_files/final_architecture2.pth.tar ' + folder + "/exp_files/final_architecture2.pth.tar")
        os.popen('cp  exp/TIMIT_LSTM_fbank_c4/exp_files/final_architecture3.pth.tar ' + folder + "/exp_files/final_architecture3.pth.tar")
        os.popen('cp  exp/TIMIT_LSTM_fbank_c4/exp_files/checkpoint_architecture1.pth.tar ' + folder + "/exp_files/checkpoint_architecture1.pth.tar")
        os.popen('cp  exp/TIMIT_LSTM_fbank_c4/exp_files/checkpoint_architecture2.pth.tar ' + folder + "/exp_files/checkpoint_architecture2.pth.tar")
        os.popen('cp  exp/TIMIT_LSTM_fbank_c4/exp_files/checkpoint_architecture3.pth.tar ' + folder + "/exp_files/checkpoint_architecture3.pth.tar")
        os.popen('c  exp/TIMIT_LSTM_fbank_c4/quantize_TIMIT.cfg  ' + folder + "/quantize_TIMIT.cfg")

    #     os.popen('cp /usr/local/home/nesrez/conda/envs/env/lib/python3.7/site-packages/sru-2.5.1-py3.7.egg/sru/modules_t.py /usr/local/home/nesrez/conda/envs/env/lib/python3.7/site-packages/sru-2.5.1-py3.7.egg/sru/modules.py')

        #if wer<20:
        #    retrain(Kaldi_path, folder, old_folder, x[0], x[1], x[2], x[3], 4, 4, 4, 4, x[8], x[9], x[10], 4, 4, 4, x[14], 16,
        #            30)
        #else:
    retrain_lstm(Kaldi_path, folder,old_folder, x,4, 10)
   #     os.popen('cp /usr/local/home/nesrez/conda/envs/env/lib/python3.7/site-packages/sru-2.5.1-py3.7.egg/sru/modules_i.py /usr/local/home/nesrez/conda/envs/env/lib/python3.7/site-packages/sru-2.5.1-py3.7.egg/sru/modules.py')

    config = configparser.ConfigParser()
    config.read(folder + '/quantize_TIMIT.cfg')
    config.set("exp", "out_folder", folder)
    config.set("architecture1","arch_pretrain_file",folder+"/exp_files/quantize_TIMIT_architecture1.pth.tar")
    config.set("architecture2","arch_pretrain_file",folder+"/exp_files/quantize_TIMIT_architecture2.pth.tar")
    config.set("architecture3","arch_pretrain_file","exp/TIMIT_LSTM_fbank_c4/exp_files/final_architecture3.pth.tar")
    config.set("architecture1","conf_file","/usr/local/home/nesrez/pytorch-kaldi/"+folder+"/quantize_TIMIT.cfg")
    with open(folder + '/quantize_TIMIT.cfg', 'w') as f:
        config.write(f)

    beacons.append(x)
    beacons_folders.append(folder)

def select_beacon( sol):
    beacon_index=-1
    min_distance=999999
    for i in range(len(beacons)):
        distance=compute_distance(sol,beacons[i])
        if distance<min_distance:
            min_distance=distance
            beacon_index=i
    return beacon_index,min_distance

def select_beacon_6( sol):
    beacon_index=-1
    min_distance=999999
    for i in range(len(beacons_6)):
        distance=compute_distance_6(sol,beacons_6[i])
        if distance<min_distance:
            min_distance=distance
            beacon_index=i
    return beacon_index,min_distance

def run_beacon(list,beacon_index,test=False,mono=False):
  #  if beacon_index==0:
  #      folder='exp/TIMIT_SRU_fbank_y/retrain_2'
  #  elif beacon_index==1:
  #      folder= 'exp/TIMIT_SRU_fbank_y/retrain_4'
  #  elif beacon_index==2:
  #      folder= 'exp/TIMIT_SRU_fbank_y/retrain_8_4'
  #  elif beacon_index==3:
  #      folder= 'exp/TIMIT_SRU_fbank_y/retrain_15'
   # print(beacon_index)
   # if beacon_index>4:
   #     folder="exp/TIMIT_SRU_fbank_y/retrain_0x"+str(beacon_index)
   # else:
    folder=beacons_folders[beacon_index]
    config_main.set("exp", "out_folder", folder)
    with open(cfg_file, 'w') as h:
        config_main.write(h)

    wer_q,s = run_inference_for_optimization(Kaldi_path, folder, list,delta=[0,0,0,0,0,0,0,0], n_layers=4, test=False,opt_index=4)


    wer_real=wer_q

    if test:
        wer_real, s = run_inference_for_optimization(Kaldi_path, folder, list, delta=[0, 0, 0, 0, 0, 0, 0, 0], n_layers=4,
                                                  test=True)

   # if mono == False and test==False:
   #     wer_q1 = run_inference_for_optimization(Kaldi_path, folder, list[0], list[1], list[2], list[3], list[4], list[5],
   #                                        list[6], list[7], list[8], list[9], list[10], list[11], list[12], list[13],
   #                                        list[14], list[15], test,1)


   #     wer_q2 = run_inference_for_optimization(Kaldi_path, folder, list[0], list[1], list[2], list[3], list[4], list[5],
   #                                        list[6], list[7], list[8], list[9], list[10], list[11], list[12], list[13],
   #                                        list[14], list[15], test,2)


   #     wer_q3 = run_inference_for_optimization(Kaldi_path, folder, list[0], list[1], list[2], list[3], list[4], list[5],
   #                                        list[6], list[7], list[8], list[9], list[10], list[11], list[12], list[13],
   #                                        list[14], list[15], test,3)
   #     wer_q=max(wer_q,wer_q1,wer_q2,wer_q3)
    config_main.set("exp", "out_folder", 'exp/TIMIT_SRU_fbank_dnn_6')
    with open(cfg_file, 'w') as h:
        config_main.write(h)
    return wer_q,wer_real
def run_beaconG(list,beacon_index,test=False,mono=False):
  #  if beacon_index==0:
  #      folder='exp/TIMIT_SRU_fbank_y/retrain_2'
  #  elif beacon_index==1:
  #      folder= 'exp/TIMIT_SRU_fbank_y/retrain_4'
  #  elif beacon_index==2:
  #      folder= 'exp/TIMIT_SRU_fbank_y/retrain_8_4'
  #  elif beacon_index==3:
  #      folder= 'exp/TIMIT_SRU_fbank_y/retrain_15'
   # print(beacon_index)
   # if beacon_index>4:
   #     folder="exp/TIMIT_SRU_fbank_y/retrain_0x"+str(beacon_index)
   # else:
    folder=beaconsG_folders[beacon_index]
    config_main.set("exp", "out_folder", folder)
    with open(cfg_file, 'w') as h:
        config_main.write(h)

    wer_q,s = run_inference_for_optimization_liGRU(Kaldi_path, folder, list,delta=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], n_layers=5, test=False,skip=1,opt_index=4)


    wer_real=wer_q

    if test:
        wer_real, s = run_inference_for_optimization_liGRU(Kaldi_path, folder, list, delta=[0,0,0,0,0,0,0, 0, 0, 0, 0, 0, 0, 0], n_layers=5,
                                                  test=True,skip=1)

   # if mono == False and test==False:
   #     wer_q1 = run_inference_for_optimization(Kaldi_path, folder, list[0], list[1], list[2], list[3], list[4], list[5],
   #                                        list[6], list[7], list[8], list[9], list[10], list[11], list[12], list[13],
   #                                        list[14], list[15], test,1)


   #     wer_q2 = run_inference_for_optimization(Kaldi_path, folder, list[0], list[1], list[2], list[3], list[4], list[5],
   #                                        list[6], list[7], list[8], list[9], list[10], list[11], list[12], list[13],
   #                                        list[14], list[15], test,2)


   #     wer_q3 = run_inference_for_optimization(Kaldi_path, folder, list[0], list[1], list[2], list[3], list[4], list[5],
   #                                        list[6], list[7], list[8], list[9], list[10], list[11], list[12], list[13],
   #                                        list[14], list[15], test,3)
   #     wer_q=max(wer_q,wer_q1,wer_q2,wer_q3)
    config_main.set("exp", "out_folder", 'exp/TIMIT_liGRU_fbank')
    with open(cfg_file, 'w') as h:
        config_main.write(h)
    return wer_q,wer_real

def run_beacon_v2(list,beacon_index,test=False,mono=False):
  #  if beacon_index==0:
  #      folder='exp/TIMIT_SRU_fbank_y/retrain_2'
  #  elif beacon_index==1:
  #      folder= 'exp/TIMIT_SRU_fbank_y/retrain_4'
  #  elif beacon_index==2:
  #      folder= 'exp/TIMIT_SRU_fbank_y/retrain_8_4'
  #  elif beacon_index==3:
  #      folder= 'exp/TIMIT_SRU_fbank_y/retrain_15'
   # print(beacon_index)
   # if beacon_index>4:
   #     folder="exp/TIMIT_SRU_fbank_y/retrain_0x"+str(beacon_index)
   # else:
    folder=beacons_folders_6[beacon_index]
    config_main.set("exp", "out_folder", folder)
    with open(cfg_file, 'w') as h:
        config_main.write(h)

    wer_q, s= run_inference_for_optimization(Kaldi_path, folder, list, delta=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],n_layers=6,test=False,opt_index=4)
  #  wer_q=20

    wer_q_test=wer_q
    if test:
        wer_q_test, s = run_inference_for_optimization(Kaldi_path, folder, list,delta=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], n_layers=6, test=True)



    #if mono == False and test==False:
    #    wer_q1 = run_inference_for_optimization(Kaldi_path, folder, list[0], list[1], list[2], list[3], list[4], list[5],
    #                                       list[6], list[7], list[8], list[9], list[10], list[11], list[12], list[13],
    #                                       list[14], list[15], test,1)


    #    wer_q2 = run_inference_for_optimization(Kaldi_path, folder, list[0], list[1], list[2], list[3], list[4], list[5],
    #                                       list[6], list[7], list[8], list[9], list[10], list[11], list[12], list[13],
    #                                       list[14], list[15], test,2)


     #   wer_q3 = run_inference_for_optimization(Kaldi_path, folder, list[0], list[1], list[2], list[3], list[4], list[5],
     #                                      list[6], list[7], list[8], list[9], list[10], list[11], list[12], list[13],
     #                                      list[14], list[15], test,3)
     #   wer_q=max(wer_q,wer_q1,wer_q2,wer_q3)
    config_main.set("exp", "out_folder", 'exp/TIMIT_SRU_fbank_dnn_l6')
    with open(cfg_file, 'w') as h:
        config_main.write(h)
    return wer_q,wer_q_test
def run_beacon_v3(list,beacon_index,test=False,mono=False):
  #  if beacon_index==0:
  #      folder='exp/TIMIT_SRU_fbank_y/retrain_2'
  #  elif beacon_index==1:
  #      folder= 'exp/TIMIT_SRU_fbank_y/retrain_4'
  #  elif beacon_index==2:
  #      folder= 'exp/TIMIT_SRU_fbank_y/retrain_8_4'
  #  elif beacon_index==3:
  #      folder= 'exp/TIMIT_SRU_fbank_y/retrain_15'
   # print(beacon_index)
   # if beacon_index>4:
   #     folder="exp/TIMIT_SRU_fbank_y/retrain_0x"+str(beacon_index)
   # else:
    folder=beacons_folders_3[beacon_index]
    config_main.set("exp", "out_folder", folder)
    with open(cfg_file, 'w') as h:
        config_main.write(h)

    wer_q, s= run_inference_for_optimization(Kaldi_path, folder, list, n_layers=3,test=test)


#    wer_q, s = run_inference_for_optimization(Kaldi_path, folder, list, n_layers=3, test=True)


    #if mono == False and test==False:
    #    wer_q1 = run_inference_for_optimization(Kaldi_path, folder, list[0], list[1], list[2], list[3], list[4], list[5],
    #                                       list[6], list[7], list[8], list[9], list[10], list[11], list[12], list[13],
    #                                       list[14], list[15], test,1)


    #    wer_q2 = run_inference_for_optimization(Kaldi_path, folder, list[0], list[1], list[2], list[3], list[4], list[5],
    #                                       list[6], list[7], list[8], list[9], list[10], list[11], list[12], list[13],
    #                                       list[14], list[15], test,2)


     #   wer_q3 = run_inference_for_optimization(Kaldi_path, folder, list[0], list[1], list[2], list[3], list[4], list[5],
     #                                      list[6], list[7], list[8], list[9], list[10], list[11], list[12], list[13],
     #                                      list[14], list[15], test,3)
     #   wer_q=max(wer_q,wer_q1,wer_q2,wer_q3)
    config_main.set("exp", "out_folder", 'exp/TIMIT_SRU_fbank_dnn_l3')
    with open(cfg_file, 'w') as h:
        config_main.write(h)
    return wer_q

#Encode the precision-bit into chromosome integer values
#2-bit-integer=> 1
#4-bit-integer=> 2
#8-bit-integer=> 3
#16-bit-fixed point=> 4

#compute speedup on bitfusion
#input the precisions of the two operands
#output: the speedup gained over 16x16 on bitfusion
def get_speedup_bitfusion(n1,n2):
    if n1*n2==256:
        s=1
    elif n1*n2 == 128:
        s=2
    elif n1*n2==64:
        s=4
    elif n1*n2==32:
        s=8
    elif n1*n2==16:

        s=16
    elif n1*n2==8:
        s=32
    elif n1*n2<=4:
        s=64

    return s

#Get the mac energy for different precision
#Silago
def get_MAC_energy(x):
    if x==16:
        return 1.666
    if x==8:
        return 1.084/2
    if x==4:
        return 0.613/4
    else: # infeasible
        return 100

#compute speedup on Silago
#input the precisions of the two operands (both operands have the same precision)
#output: the speedup gained over 16x16 on Silago
def get_speedup_silago(x):
    if x==16:
        return 1
    if x==8:
        return 2
    if x==4:
        return 4
    if x==2:
        return 8
    return 1
#write the titiles of columns in the excel sheet
def excel_init(stat):
    stat.write('B1', 'ip-to-lay0')
    stat.write('A1', 'w-lay0')
    stat.write('D1', 'ip-to-proj1')
    stat.write('C1', 'w-proj1')
    stat.write('F1', 'ip-to-lay1')
    stat.write('E1', 'W-lay1')
    stat.write('G1', 'w-proj2')
    stat.write('H1', 'ip-to-proj2')
    stat.write('I1', 'w-lay2')
    stat.write('J1', 'ip-to-lay2')
    stat.write('K1', 'w-proj3')
    stat.write('L1', 'ip-to-proj3')
    stat.write('M1', 'w-lay3')
    stat.write('N1', 'ip-to-lay3')
    stat.write('O1', 'w-fc')
    stat.write('P1', 'ip-fc')
    stat.write('Q1', 'WER')
    stat.write('R1', 'Mem_size')
    stat.write('S1' , 'real_wer')
    stat.write('T1' , 'wer_q')
    stat.write('U1' , 'real_wer_q')
    stat.write('V1' , 'distance')
#Print one solution vlaues in the excel sheet
def print_excel_delta(stat,i,F,X,real_wer):
    stat.write('B' + str(i + 2), str(X[i][1]))
    stat.write('A' + str(i + 2), str(X[i][0]))
    stat.write('D' + str(i + 2), str(X[i][3]))
    stat.write('C' + str(i + 2), str(X[i][2]))
#    stat.write('F' + str(i + 2), str(X[i][5]))
    stat.write('E' + str(i + 2), str(X[i][4]))
#    stat.write('G' + str(i + 2), str(X[i][6]))
#    stat.write('H' + str(i + 2), str(X[i][7]))

    stat.write('I' + str(i + 2), str(F[i][0]))
    stat.write('J' + str(i + 2), str(F[i][1]))
    stat.write('K' + str(i + 2), str(real_wer))

def print_excel(stat,i,F,X,real_wer,wer_q=-1,real_wer_q=-1,distance=-1):
    print(X[i])
    stat.write('B' + str(i + 2), str(X[i][4]))
    stat.write('A' + str(i + 2), str(X[i][0]))
    stat.write('D' + str(i + 2), str(X[i][11]))
    stat.write('C' + str(i + 2), str(X[i][8]))
    stat.write('F' + str(i + 2), str(X[i][5]))
    stat.write('E' + str(i + 2), str(X[i][1]))
    stat.write('G' + str(i + 2), str(X[i][9]))
    stat.write('H' + str(i + 2), str(X[i][12]))
    stat.write('I' + str(i + 2), str(X[i][2]))
    stat.write('J' + str(i + 2), str(X[i][6]))
    stat.write('K' + str(i + 2), str(X[i][10]))
    stat.write('L' + str(i + 2), str(X[i][13]))
    stat.write('M' + str(i + 2), str(X[i][3]))
    stat.write('N' + str(i + 2), str(X[i][7]))
    stat.write('O' + str(i + 2), str(X[i][14]))
    stat.write('P' + str(i + 2), str(X[i][15]))
    stat.write('Q'+str(i+2), str(F[i][0]))
    stat.write('R'+str(i+2), str(F[i][1]))
    stat.write('S'+str(i+2),str(real_wer))
    stat.write('T' + str(i + 2), str(wer_q))
    stat.write('U'+str(i+2),str(real_wer_q))
    stat.write('V' + str(i + 2), (distance))
#write the titles of the extra columns required in Silago
def excel_init_silago(stat):
    stat.write('S1', "Speedup")
    stat.write('T1' , "Energy")
    stat.write('U1', "Real_WER")

#write the titles of the extra columns required in bitfusion
def excel_init_bitfusion(stat):
    stat.write('S1', "Speedup")
    stat.write('U1', "Real_WER")
    stat.write('R1', "Mem-size")
    stat.write('T1', "WER Q")
    stat.write('W1' , "Real WER Q")
    stat.write('V1',"distance")
#write the extra parts of the solution present in bitfusion
def print_excel_bitfusion(stat,i,F,X,real_wer,mem_size,wer_q=-1,real_wer_q=-1,index=-1,distance=-1):
    print_excel(stat,i,F,X,real_wer)
    stat.write('S'+str(i+2), str(F[i][1]))
    stat.write('U'+str(i+2),str(real_wer))
    stat.write('R' + str(i + 2), str(mem_size))
    stat.write('T' + str(i + 2), str(wer_q))
    stat.write('W' + str(i + 2), str(real_wer_q))
    stat.write('V' + str(i + 2), (index))
    stat.write('X' + str(i + 2), (distance))
#write the solution values of one solution of Silago in the excel sheet
def print_excel_silago(stat,i,F,X,real_wer,mem_size):
    print(X[i])
    stat.write('B' + str(i + 2), str(X[i][0]))
    stat.write('A' + str(i + 2), str(X[i][0]))
    stat.write('D' + str(i + 2), str(X[i][4]))
    stat.write('C' + str(i + 2), str(X[i][4]))
    stat.write('F' + str(i + 2), str(X[i][1]))
    stat.write('E' + str(i + 2), str(X[i][1]))
    stat.write('G' + str(i + 2), str(X[i][5]))
    stat.write('H' + str(i + 2), str(X[i][5]))
    stat.write('I' + str(i + 2), str(X[i][2]))
    stat.write('J' + str(i + 2), str(X[i][2]))
    stat.write('K' + str(i + 2), str(X[i][6]))
    stat.write('L' + str(i + 2), str(X[i][6]))
    stat.write('M' + str(i + 2), str(X[i][3]))
    stat.write('N' + str(i + 2), str(X[i][3]))
    stat.write('O' + str(i + 2), str(X[i][7]))
    stat.write('P' + str(i + 2), str(X[i][7]))
    stat.write('Q'+str(i+2), str(F[i][0]))
    stat.write('S'+str(i+2), str(F[i][1]))
    stat.write('T' + str(i + 2), str(F[i][2]))
    stat.write('U'+str(i+2),str(real_wer))
    stat.write('R' + str(i + 2), str(mem_size))


#Compute the memory size required by given solution for the SRU model understudy
def compute_mem(x):
    return (x[0] * 75900 + x[1] * 844800 + x[2] * 844800 + x[3] * 844800 + x[8] * 281600 + x[9] * 281600 + x[
        10] * 281600 + x[14] * 2094400 + 8800 * 16)/8/1024/1024

def compute_mem_3(x):
    return (x[0] * 75900 + x[1] * 844800 + x[2] * 844800 +  x[6] * 281600 + x[7] * 281600 + x[10] * 2094400 + 8800 * 16)/8/1024/1024

def compute_mem_6(x):
    return (x[0] * 75900 + x[1] * 844800 + x[2] * 844800 + x[3] * 844800 +x[4] * 844800 + x[5] * 844800 + x[12] * 281600 + x[13] * 281600 + x[
        14] * 281600+ x[15] * 281600 + x[
        16] * 281600 + x[22] * 2094400 + 8800 * 16)/8/1024/1024






#print("hi")
#convert_to_beacon(21,[16,2,2,2,16,2,2,2,2,2,2,2,2,16,2,16],1999)
#print("bye")
#print(compute_distance([8,2,2,4,4,4,4,4,4,4,2,6,6,6],[16,2,8,8,4,2,16,4,16,4,8,2,2,16,2,4]))
#select_beacon([2,4,4,2,8,8,8,8,2,2,2,4,4,4,2,2])
#b=((compute_mem([16,4,4,2,4,2,2,2])))#*8*1024*1024*0.08) +(75900*get_MAC_energy(16))+ (844800*get_MAC_energy(4))+(844800 * get_MAC_energy(4))+(844800*get_MAC_energy(4))+(281600 * get_MAC_energy(4))+(281600 * get_MAC_energy(8))+(281600 * get_MAC_energy(4))+(2094400 * get_MAC_energy(8)))/1000000
#print(b)
#print(compute_mem((8,2,2,2,8,8,2,2)))
#x=[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]
#print(compute_mem(x))
#print((75900*get_speedup_silago(x[0])+ 844800*get_speedup_silago(x[1])+
#        844800 * get_speedup_silago( x[2])+844800*get_speedup_silago(x[3])+
#        281600 * get_speedup_silago( x[4])+281600 * get_speedup_silago( x[5])+
#        281600 * get_speedup_silago( x[6])+2094400 * get_speedup_silago( x[7]))/5637500)
#print((compute_mem(x)*8*1024*1024*0.08 + (75900*get_MAC_energy(x[0])+ 844800*get_MAC_energy(x[1])+
#        844800 * get_MAC_energy( x[2])+844800*get_MAC_energy(x[3])+
#        281600 * get_MAC_energy( x[4])+281600 * get_MAC_energy( x[5])+
#        281600 * get_MAC_energy( x[6])+2094400 * get_MAC_energy( x[7])))/1000000)

#wer, sparsity = run_inference_for_optimization_lstm_delta('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_LSTM_fbank_c2', [0,0,0,0,0],
 #
#                                                         True, 1)
#run_beacon_v2([8,2,2,2,2,2, 4,4,4,4,4,4, 2,2,2,2,2,  4,4,4,4,4,  2,16],0)
#convert_to_beacon_v2(21,[8,2,2,2,2,2, 4,4,4,4,4,4, 2,2,2,2,2,  4,4,4,4,4,  2,16],600)

#run_beacon_v2(beacon_600,0)
#run_beacon([4,2,4,2,4,4,4,4,4,2,2,4,4,4,2,16],0,mono=True)#20.9 , 20.2
#run_beacon(beacon_152,8,mono=True)# 17.7, 18.3

#run_beacon([4,2,2,2,16,2,2,2,2,2,2,4,4,4,2,4],0,mono=True) #23.8, 24.9
#run_beacon([4,2,2,2, 16,2,2,2 ,2,2,2, 4,4,4,  2,4],1,mono=True) #19.3, 19.7

#run_beacon_v2([16,2,2,4,2,2, 4,2,2,2,2,2,   2,8,2,4,16,   2,2,2,2,2,  2,16],0)
#([4,2,2,2,2,2,  16,2,4,2,2,2,  8,4,8,4,4,    2,4,4,2,4,    2,4],0)#23.2
#run_beacon_v2([4,2,2,2,2,2,  16,2,2,2,2,2,  2,2,2,2,2,    2,4,4,2,4,    2,4],2)
#run_beacon_v2([4,2,2,2,2,2,  16,2,2,2,2,2,  2,2,2,2,2,    2,4,4,2,4,    2,4],1)
#run_beacon_v2([4,2,2,2,2,2,  16,2,2,2,2,2,  2,2,2,2,2,    2,4,4,2,4,    2,4],2)
#run_beacon_v2([16,2,2,4,2,2, 4,4,4,4,4,4,   2,8,2,4,16,   4,4,4,4,4,  2,8],0)

#run_beacon_v2([16,2,2,2,2,2, 4,4,4,4,4,4,   2,8,2,4,16,   4,4,4,4,4,  2,4],0)
#run_beacon_v2([16,2,2,2,2,2, 4,4,4,4,4,4,   2,8,2,4,16,   4,4,4,4,4,  2,4],3)
#run_beacon_v2([16,2,2,2,2,2, 4,4,4,4,4,4,   2,8,2,4,16,   4,4,4,4,4,  2,4],4)


#run_beacon_v2([16,2,2,4,2,4, 4,4,4,4,4,4,   2,8,2,4,16,   4,4,4,4,4,  2,8],0)#19.2
#run_beacon_v2([16,2,2,4,2,4, 4,4,4,4,4,4,   2,8,2,4,16,   4,4,4,4,4,  2,8],4)#18

#run_beacon_v2([16,2,2,4,2,4, 4,4,4,4,4,4,   2,8,2,4,16,   4,4,4,4,4,  2,4],0)#20.9
#run_beacon_v2([16,2,2,4,2,4, 4,4,4,4,4,4,   2,8,2,4,16,   4,4,4,4,4,  2,4],4)#19.7

#run_beacon_v2([16,2,2,4,2,4, 4,4,4,4,4,4,   2,8,2,4,2,   4,4,4,4,4,  2,4],0)#21.4
#run_beacon_v2([16,2,2,4,2,4, 4,4,4,4,4,4,   2,8,2,4,2,   4,4,4,4,4,  2,4],4)#20.1


#run_beacon_v2([16,2,2,4,2,4, 4,4,4,4,4,4,   2,8,2,4,4,   4,4,4,4,4,  2,8],0) #19.2
#run_beacon_v2([16,2,2,4,2,4, 4,4,4,4,4,4,   2,8,2,4,4,   4,4,4,4,4,  2,8],4) #18


#run_beacon_v2([16,2,2,4,2,2, 4,4,4,4,4,4,   2,2,2,4,16,   4,4,4,4,4,  2,8],0)#19.3
#run_beacon_v2([16,2,2,4,2,2, 4,4,4,4,4,4,   2,2,2,4,16,   4,4,4,4,4,  2,8],4)#18.4

#run_beacon_v2([16,2,2,4,2,4, 4,4,4,4,4,4,   2,2,2,2,16,   4,4,4,4,4,  2,8],0)#19.6
#run_beacon_v2([16,2,2,4,2,4, 4,4,4,4,4,4,   2,2,2,2,16,   4,4,4,4,4,  2,8],4)#17.8

#run_beacon_v2([16,2,2,2,2,2, 4,4,4,4,4,4,   2,8,2,4,16,   4,4,4,4,4,  2,8],0)#19.7
#run_beacon_v2([16,2,2,2,2,2, 4,4,4,4,4,4,   2,8,2,4,16,   4,4,4,4,4,  2,8],4)#18.1

#run_beacon_v2([16,2,2,2,2,4, 4,4,4,4,4,4,   2,4,2,4,2,   4,4,4,4,4,  2,8],0)#19.7
#run_beacon_v2([16,2,2,2,2,4, 4,4,4,4,4,4,   2,4,2,4,2,   4,4,4,4,4,  2,8],4)#18
#run_beacon_v2([16,2,2,4,2,2, 4,4,4,4,4,4,   2,8,2,4,16,   4,4,4,4,4,  2,8],0)
#run_beacon_v2(beacon_399,3)
#run_beacon_v2([4,2,2,2,2,2,16,2,2,2,2,2,2,2,2,2,2,4,4,4,4,4,2,4],0) #23.9
#run_beacon_v2([4,2,2,2,2,2,16,2,2,2,2,2,2,2,2,2,2,4,4,4,4,4,2,4],0,mono=True) #23.2

#run_beacon_v2([4,2,2,2,4,2,  8,4,4,2,2,4,   2,4,2,4,8,    2,4,4,4,4,   2,4],2)
#run_beacon_v2([4,2,2,2,4,2,  8,4,4,2,2,4,   2,4,2,4,8,    2,4,4,4,4,   2,4],1)
#run_beacon_v2([4,2,2,2,4,2,  8,4,4,2,2,4,   2,4,2,4,8,    2,4,4,4,4,   2,4],2)
#run_beacon_v2([4,2,2,2,4,2,  8,4,4,2,2,4,   2,4,2,4,8,    2,4,4,4,4,   2,4],0)
#run_beacon_v2([4,2,2,2,2,2,16,2,2,2,2,2,2,2,2,2,2,4,4,4,4,4,2,4],9,mono=True) #20.4


#print(compute_distance_6([4,2,2,2,2,2,  16,2,4,2,2,2,  8,4,8,4,4,    2,4,4,2,4,    2,4],beacon_465))

#run_beacon_v2([4,2,2,2,2,2,  16,2,4,2,2,2,  8,4,8,4,4,    2,4,4,2,4,    2,4],6)
#run_beacon_v2([4,2,2,2,2,2,  16,2,4,2,2,2,  8,4,8,4,4,    2,4,4,2,4,    2,4],0)

#convert_to_beacon_v2(21,beacon_600,6000000)

#beacon_465=[4,2,2,2,2,2,  8,4,4,2,2,4,   8,4,8,4,8,    2,4,4,4,4,   2,4]
#beacon_600=[8,2,2,2,2,2, 4,4,4,4,4,4, 2,2,2,2,2,  4,4,4,4,4,  2,16]
#            [4,2,2,2,2,2,  16,2,2,2,2,2,  2,2,2,2,2,    2,4,4,2,4,    2,4]


#run_inference_for_optimization(Kaldi_path,'exp/TIMIT_SRU_fbank_dnn_l3',[4,2,2,4,4,4,2,2,4,4,2,4],n_layers=3, test=False)

#convert_to_beacon_v3(21,[4,2,2,4,4,4,2,2,4,4,2,4],200)

#print("hi")
#run_beaconG(beaconG_199,0)#17.9
#run_beaconG(beaconG_192,3)#17.7
#run_beaconG(beaconG_156,2)
#run_beaconG(beaconG_156,1)

#run_beaconG([4,2,2,2,2,   4,2,2,2,2,  8,2,2,4,2,   2,4,4,2,8, 2,4],0)
#run_beaconG([4,2,2,2,2,   4,2,2,2,2,  8,2,2,4,2,   2,4,4,2,8, 2,4],2)

#run_beaconG([4,2,4,2,2,   4,2,4,4,2,  4,2,2,2,2,   2,2,2,2,8, 2,4],0)
#run_beaconG([4,2,4,2,2,   4,2,4,4,2,  4,2,2,2,2,   2,2,2,2,8, 2,4],2)

#run_beaconG([4,2,16,2,2,   2,2,2,2,2,  8,2,2,4,2,   4,4,2,2,2, 2,8],0)
#run_beaconG([4,2,16,2,2,   2,2,2,2,2,  8,2,2,4,2,   4,4,2,2,2, 2,8],2)

#run_beaconG([4,2,4,2,2,   4,2,2,8,8,  8,2,2,2,2,   2,2,2,2,2, 4,4],0)
#run_beaconG([4,2,4,2,2,   4,2,2,8,8,  8,2,2,2,2,   2,2,2,2,2, 4,4],2)

#run_beaconG([4,2,2,4,2,   4,2,16,2,2,  4,2,2,4,2,   2,2,16,2,4, 2,4],0)
#run_beaconG([4,2,2,4,2,   4,2,16,2,2,  4,2,2,4,2,   2,2,16,2,4, 2,4],2)

#run_beaconG([8,4,2,2,2,   2,2,2,8,2,  4,2,2,2,4,   2,16,2,2,8, 2,4],0)
#run_beaconG([8,4,2,2,2,   2,2,2,8,2,  4,2,2,2,4,   2,16,2,2,8, 2,4],2)

#run_beaconG([4,2,2,2,2,   4,2,2,2,2,  8,2,2,2,2,   2,2,2,2,2, 2,4],0)
#run_beaconG([4,2,2,2,2,   4,2,2,2,2,  8,2,2,2,2,   2,2,2,2,2, 2,4],2)

#run_beaconG([8,4,2,2,2,   2,2,2,2,2,  8,2,2,2,4,   2,2,2,2,8, 2,4],0)
#run_beaconG([8,4,2,2,2,   2,2,2,2,2,  8,2,2,2,4,   2,2,2,2,8, 2,4],1)

#run_beaconG([16,4,2,2,2,   2,2,2,2,2,  16,2,2,2,4,   2,2,2,2,8, 2,4],0)
#run_beaconG([16,4,2,2,2,   2,2,2,2,2,  16,2,2,2,4,   2,2,2,2,8, 2,4],2)

#run_beaconG([2,2,2,2,2,   4,2,2,2,2,  8,2,2,4,2,   2,4,4,2,8, 2,4],0)
#run_beaconG([2,2,2,2,2,   4,2,2,2,2,  8,2,2,4,2,   2,4,4,2,8, 2,4],2)



################################333
#run_beaconG([16,2,2,2,2,   16,2,2,4,2,  16,2,2,2,2,   16,2,2,2,8, 2,4],0)#20.3
#run_beaconG([16,2,2,2,2,   16,2,2,4,2,  16,2,2,2,2,   16,2,2,2,8, 2,4],1)#18.1
#run_beaconG([16,2,2,2,2,   16,2,2,4,2,  16,2,2,2,2,   16,2,2,2,8, 2,4],2)#18.3

#run_beaconG([2,2,4,2,2,   16,2,2,2,2,  16,2,4,2,2,   16,2,4,2,2, 2,8],0) #28.1
#run_beaconG([2,2,4,2,2,   16,2,2,2,2,  16,2,4,2,2,   16,2,4,2,2, 2,8],1) #25.8
#run_beaconG([2,2,4,2,2,   16,2,2,2,2,  16,2,4,2,2,   16,2,4,2,2, 2,8],3) #25.5


#run_beaconG([16,2,4,2,2,   16,2,2,2,2,  16,2,4,2,2,   16,2,4,2,2, 2,8],0)#20.3
#run_beaconG([16,2,4,2,2,   16,2,2,2,2,  16,2,4,2,2,   16,2,4,2,2, 2,8],1)#18.3
#run_beaconG([16,2,4,2,2,   16,2,2,2,2,  16,2,4,2,2,   16,2,4,2,2, 2,8],2)#18.4

#run_beaconG([16,4,2,2,2,   2,4,2,2,2,  16,2,2,2,4,   4,2,2,2,2, 2,4],1)#17.2
#run_beaconG([16,4,2,2,2,   2,4,2,2,2,  16,2,2,2,4,   4,2,2,2,2, 2,4],2)#17.3

#run_beaconG([16,2,2,2,2,   2,2,2,2,2,  16,2,4,2,4,   2,2,2,2,4, 2,8],1)#18.7
#run_beaconG([16,2,2,2,2,   2,2,2,2,2,  16,2,4,2,4,   2,2,2,2,4, 2,8],2)#18.7

#run_beaconG([16,4,2,4,2,   2,2,4,2,2,  16,2,4,2,2,   2,4,2,2,2, 2,4],1)#17
#run_beaconG([16,4,2,4,2,   2,2,4,2,2,  16,2,4,2,2,   2,4,2,2,2, 2,4],2)#16.9

#run_beaconG([16,2,2,8,2,   4,2,2,2,4,  16,2,2,2,4,   2,2,2,2,2, 2,4],1)
#run_beaconG([16,2,2,8,2,   4,2,2,2,4,  16,2,2,2,4,   2,2,2,2,2, 2,4],2)


#convert_to_beacon_G(20,beaconG_192,192)
#run_beacon_v3([4,2,2,4,4,4,2,2,4,4,2,4],0)
#run_beacon_v3([4,2,2,4,4,4,2,2,4,4,2,4],1)

#run_beacon_v2([4,2,2,2,2,2,16,2,8,2,2,2,2,4,2,4,4,2,8,4,8,4,2,8],1)
#run_beacon_v2([4,2,2,2,2,2,16,2,8,2,2,2,2,4,2,4,4,2,8,4,8,4,2,8],2)
#run_beacon_v2([4,2,2,2,2,2,  16,2,2,2,2,2,  2,2,2,2,2,  2,2,2,2,2,  2,4],5)
#run_beacon_v2([4,2,2,2,2,2,  16,2,2,2,2,2,  2,2,2,2,2,  2,2,2,2,2,  2,4],6)
#run_beacon([8,2,2,2,16,4,2,2,8,8,2,2,2,8,2,4],9,mono=True)
#run_beacon([8,2,2,2,16,4,2,2,8,8,2,2,2,8,2,4],10,mono=True)
#os.popen('cp /usr/local/home/nesrez/conda/envs/env/lib/python3.7/site-packages/sru-2.5.1-py3.7.egg/sru/modules_new.py /usr/local/home/nesrez/conda/envs/env/lib/python3.7/site-packages/sru-2.5.1-py3.7.egg/sru/modules.py')

#run_beacon_v3([2,4,2,2,16,2,2,2,4,2,2,4],0,mono=True)
#run_beacon([2,4,2,2,16,2,2,2,4,2,4,2,2,2,2,4],12,mono=True)
#run_beacon([2,4,2,2,16,2,16,2,4,2,4,2,2,4,2,4],7,mono=True)

#os.popen('cp /usr/local/home/nesrez/conda/envs/env/lib/python3.7/site-packages/sru-2.5.1-py3.7.egg/sru/modules_back.py /usr/local/home/nesrez/conda/envs/env/lib/python3.7/site-packages/sru-2.5.1-py3.7.egg/sru/modules.py')

#run_beacon_v3([8,2,2,16,2,4,2,2,2,4,2,4],0,mono=True)
#run_beacon_v3([8,2,2,16,2,4,2,2,2,4,2,4],6,mono=True)

#convert_to_beacon_v2(21,beacon_192_6,192)
#convert_to_beacon_v3(21,[2,2,2,16,8,2,8,2,2,8,2,16],192)
#convert_to_beacon(21,beacon_156,156)

#4  21.3
#1  19.7
#0  24
#x=[2,2,2,2,2,  2,2,2,2,2,   2,2,2,2,2,   2,2,2,2,2,   2,4]
#print(get_speedup_bitfusion(x[0],x[10]) * 0.0019 + get_speedup_bitfusion(x[1],x[11])* 0.0927 + get_speedup_bitfusion(x[2],x[12]) * 0.0927 \
#             + get_speedup_bitfusion(x[3],x[13]) * 0.0927 + get_speedup_bitfusion(x[4],x[14])* 0.0927 + 1/2*(get_speedup_bitfusion(x[5],x[15]) * 0.0927 \
#            + get_speedup_bitfusion(x[6],x[16]) * 0.0927 + get_speedup_bitfusion(x[7],x[17]) * 0.0927 + get_speedup_bitfusion(x[8],x[18]) * 0.0927+\
#             get_speedup_bitfusion(x[9],x[19]) * 0.0927 ) + 1/2*0.0927*5+ get_speedup_bitfusion(x[20],x[21]) * 0.163 + 0.0002)
#print(21.2/compute_mem([8,2,2,16,8,2,16,16,4,2,2,4,4,16,2,4]))
#run_beacon_v2([16,2,4,2,8,2,   16,2,4,2,8,2,   8,2,4,8,2,  2,16,2,16,4,  16,4],2)
# 18 17.9 18.7  new:17.2 #16,2,4,2,8,2,   16,2,4,2,8,2,   8,2,4,8,2,  2,16,2,16,4,  16,4

#run_beacon_v2([16,2,2,2,2,2,  16,4,16,2,16,2, 8,8,2,8,2, 2,2,8,2,8,   2,4],9)#19.6  3.5
#run_beacon_v2([16,2,2,2,2,2,  16,4,16,2,16,2, 8,8,2,8,2, 2,2,8,2,8,   2,4],1)#18.1  1.5
#run_beacon_v2([16,2,2,2,2,2,  16,4,16,2,16,2, 8,8,2,8,2, 2,2,8,2,8,   2,4],2)# 17.8
#run_beacon_v2([16,2,2,2,2,2,  16,4,16,2,16,2, 8,8,2,8,2, 2,2,8,2,8,   2,4],5)
#17.4
#convert_to_beacon(21,beacon_156,156)
#19.3,18.4,18
#17.9
#19.5,19.8,18.5
#17.1->19.1,17.5->19.8, 16.4->18.5
