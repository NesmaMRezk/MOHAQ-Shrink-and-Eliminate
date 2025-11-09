import configparser
import sys

from pandas.tests.io.excel.test_xlsxwriter import xlsxwriter

from optimization_utils import compute_mem, compute_distance
from run_exp_test import run_inference_for_optimization, run_inference_for_optimization_LSTM
import os
from run_exp import retrain
#SET A
#retrain_8_4 8,8,8,8,4,4,4,4,8,8,8,4,4,4,16,16
#retrain_2  2,2,2,2,16,16,16,16,2,2,2,16,16,16,16,16,24
#retrain_4 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,16
#retrain_15 16,2,2,2,16,16,16,16,4,4,4,16,16,16,16,16
#retrain_4_2 4,4,4,4,4,4,4,4,2,2,2,16,16,16,4,16
Kaldi_path= os.environ['KALDI_ROOT']


#4,4,4,4,4,4,4,4,2,2,2,16,16,16,16,16
#retrain(Kaldi_path,'exp/TIMIT_SRU_fbank_y',4,8,2,2,16,16,4,8,16,16,2,2,2,16,16,16,20)

#run_inference_for_optimization(Kaldi_path,'exp/TIMIT_SRU_fbank_y',4,4,4,4,16,16,16,16,4,4,4,16,16,16,16,16,False)
#run_inference_for_optimization(Kaldi_path,'exp/TIMIT_SRU_fbank_y',4,4,4,4,16,16,16,16,4,4,4,16,16,16,16,16,True) #20.1
#8,4,4,4,16,16,8,16,4,4,2,16,16,16,16,16
#then I will select the best and retrain on its own
#try trianing from scratch
# find good answer for 8 bits and 4 bits
#training on 2,2,2,2,16,16,16,16,2,2,2,16,16,16,16,16,24






list_2=[[2,2,2,2,16,16,16,16,2,2,2,16,16,16,16,16],
[2,2,2,2,4,4,4,4,2,2,2,16,16,16,16,16],
[2,2,2,2,16,16,16,16,2,2,2,16,16,16,4,4],
[2,2,2,2,16,16,16,16,2,2,2,4,4,4,16,16],
[2,2,2,2,8,8,8,8,2,2,2,4,4,4,8,8],
[2,2,4,4,16,16,16,16,2,2,2,4,4,4,16,16],
[2,2,4,2,16,16,16,16,2,2,2,4,4,4,16,16],
[2,2,2,2,16,16,16,16,2,4,4,16,16,16,16,16],
[2,4,2,2,16,16,4,16,2,2,2,16,16,16,4,16],
[2,2,2,2,4,4,4,4,2,4,4,16,16,16,16,16],
[8,2,2,2,16,16,16,16,2,2,2,16,16,16,4,4],
[2,2,8,2,16,16,16,16,4,2,2,4,4,4,16,16],
[2,8,2,2,8,8,8,8,2,2,4,4,4,4,8,8],
[2,2,8,4,16,16,16,16,2,8,2,4,4,4,16,16],
[4,2,4,2,16,16,16,16,2,2,2,4,4,4,16,16],
[2,2,2,2,16,16,16,16,4,4,4,16,16,16,4,4],
]
list_15=[[16,2,2,2,16,16,16,16,4,4,4,16,16,16,16,16],
[16,2,2,2,16,16,16,16,4,4,4,16,16,16,4,4],
[16,2,2,2,16,16,16,16,4,4,4,16,16,16,8,8],
[16,2,2,2,4,4,4,4,4,4,4,16,16,16,16,16],
[16,2,2,2,16,16,16,16,4,4,4,8,8,8,16,16],
[8,2,2,2,16,16,16,16,4,4,4,8,8,8,16,16],
[16,2,2,2,8,8,8,16,4,4,4,8,8,8,16,16],
[16,2,2,2,4,4,16,16,4,4,4,16,16,16,16,16],
[16,2,2,2,8,8,8,8,4,4,4,16,16,16,16,16],
[16,2,2,2,16,16,16,16,4,4,4,4,4,4,4,4],
[16,2,2,2,16,16,16,16,4,4,4,8,8,16,8,8],
[16,2,2,2,4,4,4,4,4,4,4,16,16,16,16,16],
[4,2,2,2,16,16,16,16,4,4,4,8,8,8,16,16],
[8,2,2,2,16,16,16,16,4,4,4,8,8,8,16,16],
[2,2,2,2,8,8,8,16,4,4,4,8,8,8,16,16],
[16,2,2,2,4,4,4,4,4,4,4,16,16,16,16,16],
]
print("*************************************************************************************")
list_4=[
[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],
[4,4,4,4,4,4,4,4,4,4,4,4,8,8,4,4],
[4,4,4,4,16,16,16,16,4,4,4,4,4,4,4,4],
[4,4,4,4,4,4,4,4,4,4,4,8,8,8,4,4],
[4,4,2,2,4,4,4,4,4,4,4,4,4,4,8,8],
[4,4,4,4,4,4,4,4,4,4,4,4,2,2,4,4],
[4,8,8,4,4,4,4,4,4,4,4,4,4,4,4,4],
[4,4,4,4,4,4,4,4,4,4,8,4,8,8,4,4],
[4,4,2,4,16,16,16,16,4,4,2,4,4,4,4,4],
[4,4,4,4,4,4,4,4,4,2,2,8,8,8,4,4],
[4,4,2,2,4,4,4,4,4,4,4,2,2,4,8,8],
[8,2,4,4,4,4,4,4,4,4,4,4,2,2,4,4],
[4,4,4,4,8,8,8,4,4,4,8,4,8,8,4,4],
[4,4,2,4,16,16,16,16,4,4,2,4,4,4,4,4],
[16,4,4,4,4,4,4,4,4,2,2,8,8,4,4,4],
[4,4,2,2,4,4,4,8,16,4,2,2,2,4,8,8],
[8,2,4,4,4,8,16,4,16,4,4,8,2,2,4,4]]


list_8=[[8,8,8,8,4,4,4,4,8,8,8,4,4,4,16,16],
[8,8,8,8,4,4,4,4,8,8,8,4,4,4,4,4],
[8,8,8,8,4,4,4,4,8,8,8,4,4,4,8,8],
[8,8,8,8,8,8,8,8,8,8,8,4,4,4,16,16],
[8,8,8,8,4,4,4,4,8,8,8,8,8,8,16,16],
[8,8,8,8,8,8,8,8,8,8,8,4,4,4,16,16],
[8,8,8,4,4,4,4,4,8,8,4,4,4,4,16,16],
[8,8,8,8,4,4,4,4,4,8,8,4,4,4,4,4],
[8,4,4,8,4,4,4,4,8,8,8,4,4,4,16,16],
[8,8,8,8,4,4,4,4,8,8,8,4,4,4,4,4],
[8,8,8,8,4,4,4,4,8,2,2,4,4,4,8,8],
[8,8,8,8,8,8,8,8,8,4,4,4,4,4,16,16],
[8,2,2,8,4,4,4,4,8,8,8,8,8,8,16,16],
[8,8,4,2,8,8,8,8,8,8,8,4,4,4,16,16],
[8,8,8,4,4,4,4,4,8,8,4,4,4,4,16,16],
[16,8,8,8,4,4,4,4,4,8,8,4,4,4,4,4]
]

list_2_base=[2,2,2,2,16,16,16,16,2,2,2,16,16,16,16,16]
list_4_base=[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]
list_8_base=[8,8,8,8,4,4,4,4,8,8,8,4,4,4,16,16]
list_15_base=[16,2,2,2,16,16,16,16,4,4,4,16,16,16,16,16]

excel = xlsxwriter.Workbook('retrained_sheets_dst_abs.xlsx')
stat_2 = excel.add_worksheet()
stat_4 = excel.add_worksheet()
stat_8_4 = excel.add_worksheet()
stat_15 = excel.add_worksheet()

stat_2.write('A1', 'solution')
stat_2.write('B1', 'mem_size')
stat_2.write('C1', 'comp.ratio')
stat_2.write('D1', 'Base WER')
stat_2.write('E1', 'retrain-2 WER')
stat_2.write('F1', 'error red.')
stat_2.write('G1', '2_distance')
stat_2.write('H1', 'retrain-4 WER')
stat_2.write('I1', 'error red.')
stat_2.write('J1', '4_distance')
stat_2.write('K1', 'retrain-8 WER')
stat_2.write('L1', 'error red.')
stat_2.write('M1', '8_distance')
stat_2.write('N1', 'retrain-15 WER')
stat_2.write('O1', 'error red.')
stat_2.write('P1', '15_distance')



stat_4.write('A1', 'solution')
stat_4.write('B1', 'mem_size')
stat_4.write('C1', 'comp.ratio')
stat_4.write('D1', 'Base WER')
stat_4.write('E1', 'retrain-2 WER')
stat_4.write('F1', 'error red.')
stat_4.write('G1', '2_distance')
stat_4.write('H1', 'retrain-4 WER')
stat_4.write('I1', 'error red.')
stat_4.write('J1', '4_distance')
stat_4.write('K1', 'retrain-8 WER')
stat_4.write('L1', 'error red.')
stat_4.write('M1', '8_distance')
stat_4.write('N1', 'retrain-15 WER')
stat_4.write('O1', 'error red.')
stat_4.write('P1', '15_distance')

stat_8_4.write('A1', 'solution')
stat_8_4.write('B1', 'mem_size')
stat_8_4.write('C1', 'comp.ratio')
stat_8_4.write('D1', 'Base WER')
stat_8_4.write('E1', 'retrain-2 WER')
stat_8_4.write('F1', 'error red.')
stat_8_4.write('G1', '2_distance')
stat_8_4.write('H1', 'retrain-4 WER')
stat_8_4.write('I1', 'error red.')
stat_8_4.write('J1', '4_distance')
stat_8_4.write('K1', 'retrain-8 WER')
stat_8_4.write('L1', 'error red.')
stat_8_4.write('M1', '8_distance')
stat_8_4.write('N1', 'retrain-15 WER')
stat_8_4.write('O1', 'error red.')
stat_8_4.write('P1', '15_distance')

stat_15.write('A1', 'solution')
stat_15.write('B1', 'mem_size')
stat_15.write('C1', 'comp.ratio')
stat_15.write('D1', 'Base WER')
stat_15.write('E1', 'retrain-2 WER')
stat_15.write('F1', 'error red.')
stat_15.write('G1', '2_distance')
stat_15.write('H1', 'retrain-4 WER')
stat_15.write('I1', 'error red.')
stat_15.write('J1', '4_distance')
stat_15.write('K1', 'retrain-8 WER')
stat_15.write('L1', 'error red.')
stat_15.write('M1', '8_distance')
stat_15.write('N1', 'retrain-15 WER')
stat_15.write('O1', 'error red.')
stat_15.write('P1', '15_distance')



#stat.write('G1', 'retrain-q15 WER')
#stat.write('H1', 'error red.')
#stat.write('I1', 'retrain-q4 WER')
#stat.write('J1', 'error red.')
#stat.write('K1', 'retrain-q2 WER')
#stat.write('L1', 'error red.')
#stat.write('M1', 'retrain-4-2 WER')
#stat.write('N1', 'error red.')


cfg_file = "cfg/TIMIT_baselines/TIMIT_SRU_fbank.cfg"
if not (os.path.exists(cfg_file)):
    sys.stderr.write("ERROR: The config file %s does not exist!\n" % (cfg_file))
    sys.exit(0)
else:

    config_main = configparser.ConfigParser()
    config_main.read(cfg_file)
#config_main.set("exp", "out_folder", 'exp/TIMIT_SRU_fbank_y/retrain_0x42')
#with open(cfg_file, 'w') as h:
#    config_main.write(h)

#wer_base=run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_y/retrain_0x42',4,8,2,4,16,2,8,4,2,16,16,4,2,16,4,8,False)
#wer_base=run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_y/retrain_0x42',4,8,2,4,16,2,8,4,2,16,16,4,2,16,4,8,True)

#config_main = configparser.ConfigParser()
#config_main.read(cfg_file)
#config_main.set("exp", "out_folder", 'exp/TIMIT_SRU_fbank_y/retrain_0x1900')
#with open(cfg_file, 'w') as h:
#    config_main.write(h)

#wer_base=run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_y/retrain_0x1900',4,8,2,4,16,2,8,4,2,16,16,4,2,16,4,8,False)
#wer_base=run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_y/retrain_0x1900',4,8,2,4,16,2,8,4,2,16,16,4,2,16,4,8,True)

#config_main.set("exp", "out_folder", 'exp/retraining/retrain_v1999')
#with open(cfg_file, 'w') as h:
#    config_main.write(h)
#run_inference_for_optimization(Kaldi_path, 'exp/retraining/retrain_v1999',8,4,4,4,8,4,4,4,4,4,4,4,4,4,4,4, False)
#run_inference_for_optimization(Kaldi_path, 'exp/retraining/retrain_v1999',8,4,4,4,8,4,4,4,4,4,4,4,4,4,4,4, True)

config_main.set("exp", "out_folder", 'exp/TIMIT_SRU_fbank_dnn_oo')
with open(cfg_file, 'w') as h:
    config_main.write(h)
#run_inference_for_optimization_LSTM(Kaldi_path, 'exp/TIMIT_LSTM_fbank_c2',16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,8,False)
#run_inference_for_optimization_LSTM(Kaldi_path, 'exp/TIMIT_LSTM_fbank_c2',16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,8,True)
wer=run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_dnn_oo',16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,False)
wer_test=run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_dnn_oo',16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,True)
wer1 = run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_dnn_oo',8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8, False, 1)
wer2 = run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_dnn_oo',8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,False, 2)
wer3 = run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_dnn_oo', 8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8, False, 3)
wer = max(wer1, wer2, wer3, wer)
print(wer)
print(wer_test)
#run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_dnn_oo',8,2,4,4,16,16,16,8,4,2,4,4,8,16,4,8,True)

#run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_dnn_2',4,4,8,8,16,4,2,16,4,4,16,4,16,8,16,16,False)
#run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_dnn_2',4,4,8,8,16,4,2,16,4,4,16,4,16,8,16,16,True)

#run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_dnn_2',16,8,8,8,16,4,2,16,4,4,16,4,16,8,16,16,False)
#run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_dnn_2',16,8,8,8,16,4,2,16,4,4,16,4,16,8,16,16,True)

#run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_dnn_2',4,16,8,8,16,16,2,16,4,4,16,4,16,4,16,16,False)
#run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_dnn_2',4,16,8,8,16,16,2,16,4,4,16,4,16,4,16,16,True)

#run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_dnn_2',4,16,16,8,16,16,2,16,16,8,4,8,16,4,16,16,False)
#run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_dnn_2',4,16,16,8,16,16,2,16,16,8,4,8,16,4,16,16,True)

#run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_dnn_2',4,4,8,8,8,4,2,16,4,4,8,8,16,4,16,16,False)
#run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_dnn_2',4,4,8,8,8,4,2,16,4,4,8,8,16,4,16,16,True)

#run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_dnn_2',4,4,4,8,8,2,16,16,4,8,16,8,16,4,16,16,False)
#run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_dnn_2',4,4,4,8,8,2,16,16,4,8,16,8,16,4,16,16,True)

#run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_dnn_2',4,16,16,8,8,4,2,16,4,4,8,4,16,16,16,16,False)
#run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_dnn_2',4,16,16,8,8,4,2,16,4,4,8,4,16,16,16,16,True)

#run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_dnn_2',4,4,8,8,8,16,2,16,4,4,8,8,16,4,16,16,False)
#run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_dnn_2',4,4,8,8,8,16,2,16,4,4,8,8,16,4,16,16,True)

#run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_dnn_2',4,4,8,8,16,4,2,16,4,4,16,4,16,8,16,16,False)
#run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_dnn_2',4,4,8,8,16,4,2,16,4,4,16,4,16,8,16,16,True)

#run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_dnn_2',4,16,16,8,16,4,2,16,16,4,16,4,8,8,16,16,False)
#run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_dnn_2',4,16,16,8,16,4,2,16,16,4,16,4,8,8,16,16,True)

#run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_dnn_2',4,16,8,16,8,16,2,16,4,4,16,4,16,16,16,16,False)
#run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_dnn_2',4,16,8,16,8,16,2,16,4,4,16,4,16,16,16,16,True)



config_main.set("exp", "out_folder", 'exp/TIMIT_SRU_fbank_base')
with open(cfg_file, 'w') as h:
    config_main.write(h)
#run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_dnn',4,4,8,4,16,8,8,16,4,16,16,8,16,16,16,8,False)
#run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_base',16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16, True)


config_main.set("exp", "out_folder", 'exp/TIMIT_SRU_fbank_y/retrain_0xx10000')
with open(cfg_file, 'w') as h:
    config_main.write(h)
#run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_y/retrain_0xx10000',4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,16,False)
#run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_y/retrain_0xx19000',4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,16,True)

config_main.set("exp", "out_folder", 'exp/TIMIT_SRU_fbank_y/retrain_4')
with open(cfg_file, 'w') as h:
    config_main.write(h)
#run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_y/retrain_4',4,4,4,4,4,4,4,4,4,4,4,4,4,4,16,16,False)
#run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_y/retrain_4',4,4,4,4,4,4,4,4,4,4,4,4,4,4,16,16,True)

config_main.set("exp", "out_folder", 'exp/TIMIT_SRU_fbank_y/retrain_0xx99999')
with open(cfg_file, 'w') as h:
    config_main.write(h)
#run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_y/retrain_0xx99999',4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,16,False)
#run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_y/retrain_0xx19999',4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,16,True)

for i in range(len(list_2)):
    config_main.set("exp", "out_folder", 'exp/TIMIT_SRU_fbank_y/retrain_2')

    with open(cfg_file, 'w') as h:
        config_main.write(h)
    #wer_q2 = run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_y/retrain_2', list_2[i][0], list_2[i][1],list_2[i][2], list_2[i][3], list_2[i][4], list_2[i][5], list_2[i][6],list_2[i][7], list_2[i][8], list_2[i][9], list_2[i][10], list_2[i][11],list_2[i][12], list_2[i][13], list_2[i][14], list_2[i][15], False)

    config_main.set("exp", "out_folder", 'exp/TIMIT_SRU_fbank_y/retrain_4')

    with open(cfg_file, 'w') as h:
        config_main.write(h)

    #wer_q4 = run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_y/retrain_4', list_2[i][0], list_2[i][1],list_2[i][2], list_2[i][3], list_2[i][4], list_2[i][5], list_2[i][6],list_2[i][7], list_2[i][8], list_2[i][9], list_2[i][10], list_2[i][11],list_2[i][12], list_2[i][13], list_2[i][14], list_2[i][15], False)

    config_main.set("exp", "out_folder", 'exp/TIMIT_SRU_fbank_y/retrain_8_4')

    with open(cfg_file, 'w') as h:
        config_main.write(h)
    #wer_q8 = run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_y/retrain_8_4', list_2[i][0], list_2[i][1],list_2[i][2], list_2[i][3], list_2[i][4], list_2[i][5], list_2[i][6],list_2[i][7], list_2[i][8], list_2[i][9], list_2[i][10], list_2[i][11],list_2[i][12], list_2[i][13], list_2[i][14], list_2[i][15], False)

    config_main.set("exp", "out_folder", 'exp/TIMIT_SRU_fbank_y/retrain_15')

    with open(cfg_file, 'w') as h:
        config_main.write(h)
    #wer_q15 = run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_y/retrain_15', list_2[i][0], list_2[i][1],list_2[i][2], list_2[i][3], list_2[i][4], list_2[i][5], list_2[i][6],list_2[i][7], list_2[i][8], list_2[i][9], list_2[i][10], list_2[i][11],list_2[i][12], list_2[i][13], list_2[i][14], list_2[i][15], False)

    config_main.set("exp","out_folder",'exp/TIMIT_SRU_fbank_base')
    with open(cfg_file, 'w') as h:
        config_main.write(h)
  #  run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_base', lists[i][0],lists[i][1],lists[i][2],lists[i][3],lists[i][4],lists[i][5],lists[i][6],lists[i][7],lists[i][8],lists[i][9],lists[i][10],lists[i][11],lists[i][12],lists[i][13],lists[i][14],lists[i][15], False)
  #  wer_base=run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_base',  list_2[i][0],list_2[i][1],list_2[i][2],list_2[i][3],list_2[i][4],list_2[i][5],list_2[i][6],list_2[i][7],list_2[i][8],list_2[i][9],list_2[i][10],list_2[i][11],list_2[i][12],list_2[i][13],list_2[i][14],list_2[i][15],False)
    stat_2.write('A'+str(i+2), ','.join(str(x) for x in list_2[i]))
    stat_2.write('B'+str(i+2), str(compute_mem([list_2[i][0],list_2[i][1],list_2[i][2],list_2[i][3],list_2[i][8],list_2[i][9],list_2[i][10],list_2[i][14]])))
    stat_2.write('C'+str(i+2), str(21.2/compute_mem([list_2[i][0],list_2[i][1],list_2[i][2],list_2[i][3],list_2[i][8],list_2[i][9],list_2[i][10],list_2[i][14]])))
   # stat_2.write('D'+str(i+2), str(wer_base))
   # stat_2.write('E'+str(i+2), str(wer_q2))
    #stat_2.write('F'+str(i+2), str(wer_q2-wer_base))
    stat_2.write('G'+str(i+2), compute_distance(list_2[i],list_2_base))
    #stat_2.write('H'+str(i+2), str(wer_q4))
    #stat_2.write('I'+str(i+2), str(wer_q4-wer_base))
    stat_2.write('J'+str(i+2), compute_distance(list_2[i],list_4_base))
    #stat_2.write('K'+str(i+2), str(wer_q8))
    #stat_2.write('L'+str(i+2), str(wer_q8-wer_base))
    stat_2.write('M'+str(i+2), compute_distance(list_2[i],list_8_base))
    #stat_2.write('N'+str(i+2), str(wer_q15))
    #stat_2.write('O'+str(i+2),str(wer_q15-wer_base))
    stat_2.write('P'+str(i+2), compute_distance(list_2[i],list_15_base))


for i in range(len(list_4)):
    print(i)
    config_main.set("exp", "out_folder", 'exp/TIMIT_SRU_fbank_y/retrain_2')

    with open(cfg_file, 'w') as h:
        config_main.write(h)

    #wer_q2=run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_y/retrain_2',  list_4[i][0],list_4[i][1],list_4[i][2],list_4[i][3],list_4[i][4],list_4[i][5],list_4[i][6],list_4[i][7],list_4[i][8],list_4[i][9],list_4[i][10],list_4[i][11],list_4[i][12],list_4[i][13],list_4[i][14],list_4[i][15],False)
    config_main.set("exp", "out_folder", 'exp/TIMIT_SRU_fbank_y/retrain_4')

    with open(cfg_file, 'w') as h:
        config_main.write(h)

    #wer_q4 = run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_y/retrain_4', list_4[i][0], list_4[i][1],  list_4[i][2], list_4[i][3], list_4[i][4], list_4[i][5], list_4[i][6],list_4[i][7], list_4[i][8], list_4[i][9], list_4[i][10], list_4[i][11],list_4[i][12], list_4[i][13], list_4[i][14], list_4[i][15], False)
    config_main.set("exp", "out_folder", 'exp/TIMIT_SRU_fbank_y/retrain_8_4')

    with open(cfg_file, 'w') as h:
        config_main.write(h)

    #wer_q8 = run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_y/retrain_8_4', list_4[i][0], list_4[i][1],   list_4[i][2], list_4[i][3], list_4[i][4], list_4[i][5], list_4[i][6],list_4[i][7], list_4[i][8], list_4[i][9], list_4[i][10], list_4[i][11],list_4[i][12], list_4[i][13], list_4[i][14], list_4[i][15], False)
    config_main.set("exp", "out_folder", 'exp/TIMIT_SRU_fbank_y/retrain_15')

    with open(cfg_file, 'w') as h:
        config_main.write(h)

    #wer_q15 = run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_y/retrain_15', list_4[i][0], list_4[i][1],list_4[i][2], list_4[i][3], list_4[i][4], list_4[i][5], list_4[i][6],list_4[i][7], list_4[i][8], list_4[i][9], list_4[i][10], list_4[i][11],list_4[i][12], list_4[i][13], list_4[i][14], list_4[i][15], False)




    with open(cfg_file, 'w') as h:
        config_main.write(h)
    config_main.set("exp","out_folder",'exp/TIMIT_SRU_fbank_base')
    with open(cfg_file, 'w') as h:
        config_main.write(h)
    #wer_base=run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_base',  list_4[i][0],list_4[i][1],list_4[i][2],list_4[i][3],list_4[i][4],list_4[i][5],list_4[i][6],list_4[i][7],list_4[i][8],list_4[i][9],list_4[i][10],list_4[i][11],list_4[i][12],list_4[i][13],list_4[i][14],list_4[i][15],False)
    stat_4.write('A'+str(i+2), ','.join(str(x) for x in list_4[i]))
    stat_4.write('B'+str(i+2), str(compute_mem([list_4[i][0],list_4[i][1],list_4[i][2],list_4[i][3],list_4[i][8],list_4[i][9],list_4[i][10],list_4[i][14]])))
    stat_4.write('C'+str(i+2), str(21.2/compute_mem([list_4[i][0],list_4[i][1],list_4[i][2],list_4[i][3],list_4[i][8],list_4[i][9],list_4[i][10],list_4[i][14]])))
    #stat_4.write('D'+str(i+2), str(wer_base))
    #stat_4.write('E'+str(i+2), str(wer_q2))
    #stat_4.write('F'+str(i+2), str(wer_q2-wer_base))
    stat_4.write('G'+str(i+2), compute_distance(list_4[i],list_2_base))
    #stat_4.write('H'+str(i+2), str(wer_q4))
    #stat_4.write('I'+str(i+2), str(wer_q4-wer_base))
    stat_4.write('J'+str(i+2), compute_distance(list_4[i],list_4_base))
    #stat_4.write('K'+str(i+2), str(wer_q8))
    #stat_4.write('L'+str(i+2), str(wer_q8-wer_base))
    stat_4.write('M'+str(i+2), compute_distance(list_4[i],list_8_base))
    #stat_4.write('N'+str(i+2), str(wer_q15))
    #stat_4.write('O'+str(i+2),str(wer_q15-wer_base))
    stat_4.write('P'+str(i+2), compute_distance(list_4[i],list_15_base))




for i in range(len(list_15)):
    config_main.set("exp", "out_folder", 'exp/TIMIT_SRU_fbank_y/retrain_2')
    with open(cfg_file, 'w') as h:
        config_main.write(h)
    #wer_q2 = run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_y/retrain_2', list_15[i][0], list_15[i][1],list_15[i][2], list_15[i][3], list_15[i][4], list_15[i][5], list_15[i][6],list_15[i][7], list_15[i][8], list_15[i][9], list_15[i][10], list_15[i][11],list_15[i][12], list_15[i][13], list_15[i][14], list_15[i][15], False)
    config_main.set("exp", "out_folder", 'exp/TIMIT_SRU_fbank_y/retrain_4')
    with open(cfg_file, 'w') as h:
        config_main.write(h)
    #wer_q4 = run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_y/retrain_4', list_15[i][0], list_15[i][1],list_15[i][2], list_15[i][3], list_15[i][4], list_15[i][5], list_15[i][6],list_15[i][7], list_15[i][8], list_15[i][9], list_15[i][10], list_15[i][11],list_15[i][12], list_15[i][13], list_15[i][14], list_15[i][15], False)
    config_main.set("exp", "out_folder", 'exp/TIMIT_SRU_fbank_y/retrain_8_4')
    with open(cfg_file, 'w') as h:
        config_main.write(h)
    #wer_q8 = run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_y/retrain_8_4', list_15[i][0], list_15[i][1],list_15[i][2], list_15[i][3], list_15[i][4], list_15[i][5], list_15[i][6],list_15[i][7], list_15[i][8], list_15[i][9], list_15[i][10], list_15[i][11],list_15[i][12], list_15[i][13], list_15[i][14], list_15[i][15], False)
    config_main.set("exp", "out_folder", 'exp/TIMIT_SRU_fbank_y/retrain_15')
    with open(cfg_file, 'w') as h:
        config_main.write(h)
    #wer_q15 = run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_y/retrain_15', list_15[i][0], list_15[i][1],list_15[i][2], list_15[i][3], list_15[i][4], list_15[i][5], list_15[i][6],list_15[i][7], list_15[i][8], list_15[i][9], list_15[i][10], list_15[i][11],list_15[i][12], list_15[i][13], list_15[i][14], list_15[i][15], False)

    config_main.set("exp","out_folder",'exp/TIMIT_SRU_fbank_base')
    with open(cfg_file, 'w') as h:
        config_main.write(h)
   # wer_base=run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_base',  list_15[i][0],list_15[i][1],list_15[i][2],list_15[i][3],list_15[i][4],list_15[i][5],list_15[i][6],list_15[i][7],list_15[i][8],list_15[i][9],list_15[i][10],list_15[i][11],list_15[i][12],list_15[i][13],list_15[i][14],list_15[i][15],False)
    stat_15.write('A'+str(i+2), ','.join(str(x) for x in list_15[i]))
    stat_15.write('B'+str(i+2), str(compute_mem([list_15[i][0],list_15[i][1],list_15[i][2],list_15[i][3],list_15[i][8],list_15[i][9],list_15[i][10],list_15[i][14]])))
    stat_15.write('C'+str(i+2), str(21.2/compute_mem([list_15[i][0],list_15[i][1],list_15[i][2],list_15[i][3],list_15[i][8],list_15[i][9],list_15[i][10],list_15[i][14]])))
    #stat_15.write('D'+str(i+2), str(wer_base))
    #stat_15.write('E' + str(i + 2), str(wer_q2))
    #stat_15.write('F' + str(i + 2), str(wer_q2 - wer_base))
    stat_15.write('G' + str(i + 2), compute_distance(list_15[i], list_2_base))
    #stat_15.write('H' + str(i + 2), str(wer_q4))
    #stat_15.write('I' + str(i + 2), str(wer_q4 - wer_base))
    stat_15.write('J' + str(i + 2), compute_distance(list_15[i], list_4_base))
    #stat_15.write('K' + str(i + 2), str(wer_q8))
    #stat_15.write('L' + str(i + 2), str(wer_q8 - wer_base))
    stat_15.write('M' + str(i + 2), compute_distance(list_15[i], list_8_base))
    #stat_15.write('N' + str(i + 2), str(wer_q15))
    #stat_15.write('O' + str(i + 2), str(wer_q15 - wer_base))
    stat_15.write('P' + str(i + 2), compute_distance(list_15[i], list_15_base))



for i in range(len(list_8)):
    config_main.set("exp", "out_folder", 'exp/TIMIT_SRU_fbank_y/retrain_2')
    with open(cfg_file, 'w') as h:
        config_main.write(h)
    print(i)
    #wer_q2 = run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_y/retrain_2', list_8[i][0], list_8[i][1],list_8[i][2], list_8[i][3], list_8[i][4], list_8[i][5], list_8[i][6],list_8[i][7], list_8[i][8], list_8[i][9], list_8[i][10], list_8[i][11],list_8[i][12], list_8[i][13], list_8[i][14], list_8[i][15], False)
    config_main.set("exp", "out_folder", 'exp/TIMIT_SRU_fbank_y/retrain_4')
    with open(cfg_file, 'w') as h:
        config_main.write(h)
    #wer_q4 = run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_y/retrain_4', list_8[i][0], list_8[i][1],list_8[i][2], list_8[i][3], list_8[i][4], list_8[i][5], list_8[i][6],list_8[i][7], list_8[i][8], list_8[i][9], list_8[i][10], list_8[i][11],list_8[i][12], list_8[i][13], list_8[i][14], list_8[i][15], False)
    config_main.set("exp", "out_folder", 'exp/TIMIT_SRU_fbank_y/retrain_8_4')
    with open(cfg_file, 'w') as h:
        config_main.write(h)
    #wer_q8 = run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_y/retrain_8_4', list_8[i][0], list_8[i][1],list_8[i][2], list_8[i][3], list_8[i][4], list_8[i][5], list_8[i][6],list_8[i][7], list_8[i][8], list_8[i][9], list_8[i][10], list_8[i][11],list_8[i][12], list_8[i][13], list_8[i][14], list_8[i][15], False)
    config_main.set("exp", "out_folder", 'exp/TIMIT_SRU_fbank_y/retrain_15')
    with open(cfg_file, 'w') as h:
        config_main.write(h)
    #wer_q15 = run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_y/retrain_15', list_8[i][0], list_8[i][1],list_8[i][2], list_8[i][3], list_8[i][4], list_8[i][5], list_8[i][6],list_8[i][7], list_8[i][8], list_8[i][9], list_8[i][10], list_8[i][11],list_8[i][12], list_8[i][13], list_8[i][14], list_8[i][15], False)



    config_main.set("exp","out_folder",'exp/TIMIT_SRU_fbank_base')
    with open(cfg_file, 'w') as h:
        config_main.write(h)
  #  run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_base', list_8[i][0],lists[i][1],lists[i][2],lists[i][3],lists[i][4],lists[i][5],lists[i][6],lists[i][7],lists[i][8],lists[i][9],lists[i][10],lists[i][11],lists[i][12],lists[i][13],lists[i][14],lists[i][15], False)
    print(i)
    #wer_base=run_inference_for_optimization(Kaldi_path, 'exp/TIMIT_SRU_fbank_base',  list_8[i][0],list_8[i][1],list_8[i][2],list_8[i][3],list_8[i][4],list_8[i][5],list_8[i][6],list_8[i][7],list_8[i][8],list_8[i][9],list_8[i][10],list_8[i][11],list_8[i][12],list_8[i][13],list_8[i][14],list_8[i][15],False)
    print(i)
    stat_8_4.write('A'+str(i+2), ','.join(str(x) for x in list_8[i]))
    stat_8_4.write('B'+str(i+2), str(compute_mem([list_8[i][0],list_8[i][1],list_8[i][2],list_8[i][3],list_8[i][8],list_8[i][9],list_8[i][10],list_8[i][14]])))
    stat_8_4.write('C'+str(i+2), str(21.2/compute_mem([list_8[i][0],list_8[i][1],list_8[i][2],list_8[i][3],list_8[i][8],list_8[i][9],list_8[i][10],list_8[i][14]])))
  #  stat_8_4.write('D'+str(i+2), str(wer_base))
  #  stat_8_4.write('E' + str(i + 2), str(wer_q2))
   # stat_8_4.write('F' + str(i + 2), str(wer_q2 - wer_base))
    stat_8_4.write('G' + str(i + 2), compute_distance(list_8[i], list_2_base))
   # stat_8_4.write('H' + str(i + 2), str(wer_q4))
   # stat_8_4.write('I' + str(i + 2), str(wer_q4 - wer_base))
    stat_8_4.write('J' + str(i + 2), compute_distance(list_8[i], list_4_base))
   # stat_8_4.write('K' + str(i + 2), str(wer_q8))
   # stat_8_4.write('L' + str(i + 2), str(wer_q8 - wer_base))
    stat_8_4.write('M' + str(i + 2), compute_distance(list_8[i], list_8_base))
   # stat_8_4.write('N' + str(i + 2), str(wer_q15))
   # stat_8_4.write('O' + str(i + 2), str(wer_q15 - wer_base))
    stat_8_4.write('P' + str(i + 2), compute_distance(list_8[i], list_15_base))

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
