import os

import numpy as np
from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
from pymoo.model.problem import Problem
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.visualization.scatter import Scatter

from run_exp_test import run_inference_for_optimization
from optimization_utils import *
from optimization_utils import encode

#Nesma
#In this file optimization is done with two objectives to minimize: memory and WER
#Also WER<26 is used as a constraint to gurantee that the test wer is not > 27%
# Each solution has 16 chromosome. They represent the precisions of the weight and activation of 8 layers
# The value of each chromosome is between 1 and 4 that are the encoding of the values 2-bit-integer, 4-bit-integer, 8-bit-integer, 16-bit-fixed-point
#The initial generation has 40 population, and each generation generates 10 new
# running for 60 generations would evaluate 630 solutions
# the size of the search space is 65536 solutions
# I change the value of the last chromosome before evaluation as they give very bad solution that make the decoding very slow
#The output solutions is plotted in an png file and saved in an excel sheet file
#The output solutions are re-evaluated using the test set to know the testing WER that is also saved in the excel sheet.
#To compute the test WER. the inference has to run first with the optimization set to compute statistics required during running the testing inference.
#TODO change the out_folder value if required

Kaldi_path= os.environ['KALDI_ROOT']
#gen1_wer=[54.4,24.8,19.7,52.5,21.3,22.9,22.5,22.3,25.9,23.9,50.1,25.3,51.7,57.2,20.0,21.4,21.6,49.8,21.0,26.1,22.2,17.7,17.2,52.2,42.4,20.4,57.9,19.5,29.1,20.5,23.9,64.0,26.3,22.0,22.5,23.0,19.9,64.0,21.7,27.8]
gen1_wer=[46.0,20.3,16.6,44.6,19.8,18.9,19.5,18.7,24.0,21.4,44.3,20.0,41.7,44.3,17.7,17.8,18.4,39.9,18.2,24.0,19.2,16.5,16.3,39.7,40.1,17.0,47.6,16.9,23.6,18.0,20.3,50.5,22.6,20.2,19.7,21.1,17.8,53.4,18.6,19.2]
distance_threshold=6
class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=24, n_obj=2, n_constr=1, xl=1, xu=4, type_var=np.int,elementwise_evaluation=True)
        self.count=0
    def _evaluate(self, x  , out, *args, **kwargs):

        print(self.count)
        for i in range(24):
            x[i]=encode(x[i])
        #TODO otherwise decoding is very slow
        if x[23]==2:
            x[23]=4
 #       if x[15]<x[14]:
 #           x[15]*=2
        f2=compute_mem_6(x)

        #invalid solutions, activations < weight
        #if x[1]<x[0] or x[3]<x[2] or x[5]<x[4] or x[7]<x[6] or x[9]<x[8] or x[11]<x[10] or x[13]<x[12]:
        #    wer=100
        #else:
      #  if self.count<40:
      #      wer=gen1_wer[self.count]
      #  else:
        wer,s=run_inference_for_optimization('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_SRU_fbank_dnn_l6', x,delta=[0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0], n_layers=6, test=False,opt_index=4)
        wer_f, s = run_inference_for_optimization('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_SRU_fbank_dnn_l6', x,
                                                delta=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], n_layers=6, test=False,
                                                opt_index=5)
        wer_t,s=run_inference_for_optimization('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_SRU_fbank_dnn_l6', x,delta=[0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0], n_layers=6, test=True)

        f1 =wer
        sheet_log.write("A" + str(self.count + 2), ','.join(str(z) for z in x))
        sheet_log.write("B" + str(self.count + 2), str(wer))
        sheet_log.write("C" + str(self.count + 2), str(wer_t))
        sheet_log.write("D" + str(self.count + 2), str(wer_f))
        sheet_log.write("E" + str(self.count + 2), str(f2))
        sheet_log.write("F" + str(self.count + 2), str(21.2/f2))
        self.count += 1


        g1=(wer-24)
        g2=(f2-6)

        out["F"] = np.column_stack([f1,f2])
        out["G"]=np.column_stack([g1])

excel = xlsxwriter.Workbook('optimize_acc_mem_dnnl6_new2.xlsx')
excel_log = xlsxwriter.Workbook('log_acc__mem_dnnl6_new2.xlsx')

sheet_log=excel_log.add_worksheet()
sheet_log.write('A1', 'solution')
sheet_log.write('B1', 'wer')
sheet_log.write('C1', 'wer_retrained')
sheet_log.write('D1', 'index')
sheet_log.write('E1', 'distance')
stat = excel.add_worksheet()

method = NSGA2(
    pop_size=40,
    n_offsprings=10,
    sampling=get_sampling("int_random"),
    crossover=get_crossover("int_sbx", prob=1.0, eta=3.0),
    mutation=get_mutation("int_pm", eta=3.0),
    eliminate_duplicates=True
)


problem = MyProblem()
#method = np.load("checkpoint.npy", allow_pickle=True).flatten()[0]


#for i in range(6):
 #   if i>-1:
method = np.load("checkpoint_acc_mem_dnn_l6_new.npy", allow_pickle=True).flatten()[0]
method.has_terminated = False
res = minimize(problem,
               method,
               ('n_gen', 100),
               seed=1,
               copy_algorithm=False,
               verbose=True)
np.save("checkpoint_acc_mem_dnn_l6_new_2", method)
excel_log.close()
plot = Scatter()
plot.add(res.F, color="red")
plot.save("pareto_acc_mem_dnn_l6_new22")
excel_init(stat)

for i in range (len(res.F)):
        # TODO otherwise decoding is very slow
    for j in range(24):
        res.X[i][j]=encode(res.X[i][j])
    if res.X[i][23] == 2:
        res.X[i][23] = 4
 #   if res.X[i][15] < res.X[i][14]:
 #       res.X[i][15] *= 2

 #   wer5 = run_inference_for_optimization('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_SRU_fbank_dnn_oo', res.X[i][0],
 #                                            res.X[i][1], res.X[i][2], res.X[i][3], res.X[i][4], res.X[i][5],
 #                                            res.X[i][6], res.X[i][7], res.X[i][8], res.X[i][9], res.X[i][10],
 #                                            res.X[i][11], res.X[i][12],
 #                                            res.X[i][13], res.X[i][14], res.X[i][15], False,4)
   # wer2 = run_inference_for_optimization('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_SRU_fbank_dnn_oo', res.X[i][0],
   #                                          res.X[i][1], res.X[i][2], res.X[i][3], res.X[i][4], res.X[i][5],
   #                                          res.X[i][6], res.X[i][7], res.X[i][8], res.X[i][9], res.X[i][10],
   #                                          res.X[i][11], res.X[i][12],
   #                                          res.X[i][13], res.X[i][14], res.X[i][15], False,2)
  #  wer3 = run_inference_for_optimization('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_SRU_fbank_dnn_oo', res.X[i][0],
  #                                           res.X[i][1], res.X[i][2], res.X[i][3], res.X[i][4], res.X[i][5],
  #                                           res.X[i][6], res.X[i][7], res.X[i][8], res.X[i][9], res.X[i][10],
  #                                           res.X[i][11], res.X[i][12],
  #                                           res.X[i][13], res.X[i][14], res.X[i][15], False,3)
 #   wer, s = run_inference_for_optimization('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_SRU_fbank_dnn_l6', res.X[i],delta=[0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0], n_layers=6, test=False, opt_index=4)
    wer_t, s = run_inference_for_optimization('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_SRU_fbank_dnn_l6', res.X[i],delta=[0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0], n_layers=6, test=True)
        #  if wer < 30:
  #      beacon_index,b = select_beacon(res.X[i])
  #      print(beacon_index)
  #      wer_q=run_beacon(res.X[i], beacon_index)
  #      real_wer_q =  run_beacon(res.X[i], beacon_index,True)
  #      distance=compute_distance(res.X[i],beacons[beacon_index])
  #      print_excel(stat,i,res.F, res.X,real_wer,wer_q,real_wer_q,str(beacon_index))
  #  else:
  #      real_wer_q=-1
  #      wer_q=-1
  #      distance=-1
    print_excel(stat, i, res.F, res.X, wer_t, 0,0, str(0))


    print("done")
excel.close()


pu