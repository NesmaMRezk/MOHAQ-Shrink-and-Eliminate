import configparser
import sys

from pandas.tests.io.excel.test_xlsxwriter import xlsxwriter

from optimization_utils import compute_mem
from run_exp_test import run_inference_for_optimization
import os
from run_exp import retrain

Kaldi_path= os.environ['KALDI_ROOT']
retrain(Kaldi_path,'exp/TIMIT_SRU_fbank_y',2,2,2,2,16,16,16,16,2,2,2,16,16,16,16,16,24)
#run_inference_for_optimization(Kaldi_path,'exp/TIMIT_SRU_fbank_y',4,4,4,4,16,16,16,16,4,4,4,16,16,16,16,16,False)
#run_inference_for_optimization(Kaldi_path,'exp/TIMIT_SRU_fbank_y',4,4,4,4,16,16,16,16,4,4,4,16,16,16,16,16,True) #20.1
#8,4,4,4,16,16,8,16,4,4,2,16,16,16,16,16

lists=[[4,4,4,4,16,16,16,16,4,4,4,16,16,16,16,16],
       [16,2,2,2,16,16,16,16,4,4,4,16,16,16,16,16],
       [16,2,2,2,16,16,16,16,4,4,4,16,16,16,8,16],
       [16,2,2,2,16,16,16,16,4,4,4,16,16,16,4,16],
       [16,2,2,2,16,16,16,16,4,4,4,16,16,16,2,16],
       [16,2,2,2,16,16,16,16,2,2,2,16,16,16,8,16],
       [16,2,2,2,16,16,16,16,2,2,2,16,16,16,2,16],
       [16,2,2,2,16,16,8,8,2,2,2,16,16,16,2,16],
       [16,2,2,2,16,4,4,4,2,2,2,8,8,8,2,16],
       [16,2,2,2,16,4,4,4,2,2,2,4,4,4,2,16],
       [2,2,2,2,8,4,4,4,2,2,2,16,16,16,2,8],
       [4,4,4,4,4,4,4,4,4,4,4,16,16,16,16,16],
       [4,4,4,4,4,4,4,4,4,4,4,16,16,16,8,8],
       [4,4,4,4,4,4,4,4,4,4,4,4,4,4,8,8],
       [8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],
       [8,8,8,8,16,16,16,16,8,8,8,16,16,16,16,16],
       [8,8,8,8,16,16,16,16,8,8,8,16,16,16,8,16],
       [8,8,8,8,16,16,16,16,8,8,8,16,16,16,4,16],
       [2,2,2,2,16,16,16,16,2,2,2,16,16,16,16,16],
       [2,2,2,2,16,16,16,16,2,2,2,16,16,16,2,16],
       [2,2,2,2,8,8,8,8,2,2,2,16,16,16,8,8],
       [2,2,2,2,8,8,8,8,2,2,2,8,8,8,8,8]]


excel = xlsxwriter.Workbook('retrained_scratch.xlsx')
stat = excel.add_worksheet()
stat.write('A1', 'solution')
stat.write('B1', 'mem_size')
stat.write('C1', 'comp.ratio')
stat.write('D1', 'Base WER')
stat.write('E1', 'retrain WER')
stat.write('F1', 'error red.')
#stat.write('G1', 'retrain-q15 WER')
#stat.write('H1', 'error red.')
#stat.write('I1', 'retrain-q2 WER')
#stat.write('J1', 'error red.')


cfg_file = "cfg/TIMIT_baselines/TIMIT_SRU_fbank.cfg"
if not (os.path.exists(cfg_file)):
    sys.stderr.write("ERROR: The config file %s does not exist!\n" % (cfg_file))
    sys.exit(0)
else:

    config_main = configparser.ConfigParser()
    config_main.read(cfg_file)


for i in range(len(lists)):
    config_main.set("exp", "out_folder", 'exp/TIMIT_SRU_fbank_y/retrain')

    with open(cfg_file, 'w') as h:
        config_main.write(h)

    run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_y/retrain', lists[i][0], lists[i][1], lists[i][2],
                                   lists[i][3], lists[i][4], lists[i][5], lists[i][6], lists[i][7], lists[i][8],
                                   lists[i][9], lists[i][10], lists[i][11], lists[i][12], lists[i][13], lists[i][14],
                                   lists[i][15], False)
    wer = run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_y/retrain', lists[i][0], lists[i][1],
                                           lists[i][2], lists[i][3], lists[i][4], lists[i][5], lists[i][6], lists[i][7],
                                           lists[i][8], lists[i][9], lists[i][10], lists[i][11], lists[i][12],
                                           lists[i][13], lists[i][14], lists[i][15], True)

    #config_main.set("exp", "out_folder", 'exp/TIMIT_SRU_fbank_y/retrain_4')

    #with open(cfg_file, 'w') as h:
    #    config_main.write(h)

    #run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_y/retrain_4', lists[i][0],lists[i][1],lists[i][2],lists[i][3],lists[i][4],lists[i][5],lists[i][6],lists[i][7],lists[i][8],lists[i][9],lists[i][10],lists[i][11],lists[i][12],lists[i][13],lists[i][14],lists[i][15], False)
    #wer_4=run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_y/retrain_4',  lists[i][0],lists[i][1],lists[i][2],lists[i][3],lists[i][4],lists[i][5],lists[i][6],lists[i][7],lists[i][8],lists[i][9],lists[i][10],lists[i][11],lists[i][12],lists[i][13],lists[i][14],lists[i][15],True)

    #config_main.set("exp", "out_folder", 'exp/TIMIT_SRU_fbank_y/retrain_15')
    #with open(cfg_file, 'w') as h:
    #    config_main.write(h)
    #run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_y/retrain_15', lists[i][0],lists[i][1],lists[i][2],lists[i][3],lists[i][4],lists[i][5],lists[i][6],lists[i][7],lists[i][8],lists[i][9],lists[i][10],lists[i][11],lists[i][12],lists[i][13],lists[i][14],lists[i][15], False)
    #wer_q15 = run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_y/retrain_15', lists[i][0], lists[i][1],
     #                                         lists[i][2], lists[i][3], lists[i][4], lists[i][5], lists[i][6],
     #                                         lists[i][7], lists[i][8], lists[i][9], lists[i][10], lists[i][11],
     #                                         lists[i][12], lists[i][13], lists[i][14], lists[i][15], True)

    #config_main.set("exp", "out_folder", 'exp/TIMIT_SRU_fbank_y/retrain_2')
    #with open(cfg_file, 'w') as h:
    #    config_main.write(h)
    #run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_y/retrain_2', lists[i][0], lists[i][1],
    #                               lists[i][2], lists[i][3], lists[i][4], lists[i][5], lists[i][6], lists[i][7],
     #                              lists[i][8], lists[i][9], lists[i][10], lists[i][11], lists[i][12], lists[i][13],
    #                               lists[i][14], lists[i][15], False)
   # wer_q2 = run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_y/retrain_2', lists[i][0], lists[i][1],
   #                                          lists[i][2], lists[i][3], lists[i][4], lists[i][5], lists[i][6],
   #                                          lists[i][7], lists[i][8], lists[i][9], lists[i][10], lists[i][11],
   #                                          lists[i][12], lists[i][13], lists[i][14], lists[i][15], True)

    config_main.set("exp","out_folder",'exp/TIMIT_SRU_fbank_base')
    with open(cfg_file, 'w') as h:
        config_main.write(h)
    run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_base', lists[i][0],lists[i][1],lists[i][2],lists[i][3],lists[i][4],lists[i][5],lists[i][6],lists[i][7],lists[i][8],lists[i][9],lists[i][10],lists[i][11],lists[i][12],lists[i][13],lists[i][14],lists[i][15], False)
    wer_base=run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_base',  lists[i][0],lists[i][1],lists[i][2],lists[i][3],lists[i][4],lists[i][5],lists[i][6],lists[i][7],lists[i][8],lists[i][9],lists[i][10],lists[i][11],lists[i][12],lists[i][13],lists[i][14],lists[i][15],True)

    stat.write('A'+str(i+2), ','.join(str(x) for x in lists[i]))
    stat.write('B'+str(i+2), str(compute_mem([lists[i][0],lists[i][1],lists[i][2],lists[i][3],lists[i][8],lists[i][9],lists[i][10],lists[i][14]])))
    stat.write('C'+str(i+2), str(21.2/compute_mem([lists[i][0],lists[i][1],lists[i][2],lists[i][3],lists[i][8],lists[i][9],lists[i][10],lists[i][14]])))
    stat.write('D'+str(i+2), str(wer_base))
    stat.write('E'+str(i+2), str(wer))
    stat.write('F'+str(i+2), str(wer_base-wer))
    #stat.write('G'+str(i+2), str(wer_q15))
    #stat.write('H'+str(i+2), str(wer_base-wer_q15))
    #stat.write('I'+str(i+2), str(wer_q2))
    #stat.write('J'+str(i+2), str(wer_base-wer_q2))
excel.close()
#retrained on for 20 epochs 16,2,2,2,16,16,16,16,4,4,4,16,16,16,16,16 in folder retrain-q1
#16,2,2,2,16,16,16,16,4,4,4,16,16,16,16,16  19.2  18.1    1.1
#16,2,2,2,16,16,16,16,4,4,4,16,16,16,8,16   20.3  18.7    1.6
#16,2,2,2,16,16,16,16,4,4,4,16,16,16,4,16   20     18.9    1.1
#16,2,2,2,16,16,16,16,4,4,4,16,16,16,2,16   20.8   19.4  (20 only using itself for retraining)  1.4

#16,2,2,2,16,16,16,16,2,2,2,16,16,16,8,16   20.6  19.2    1.4
# 16,2,2,2,16,16,16,16,2,2,2,16,16,16,2,16  21.4  19.9   1.5
# 16,2,2,2,16,16,8,8,2,2,2,16,16,16,2,16    21.6  20.1   1.5
# 16,2,2,2,16,16,8,8,2,2,2,8,8,8,2,16       21.8  20.3  1.5
# 16,2,2,2,16,4,4,4,2,2,2,8,8,8,2,16      22.3  20.5   1.8
#16,2,2,2,16,4,4,4,2,2,2,4,4,4,2,16     57.5   51.2    6.3
#2,2,2,2,8,4,4,4,2,2,2,16,16,16,2,8    24.1  21.9     2.2
