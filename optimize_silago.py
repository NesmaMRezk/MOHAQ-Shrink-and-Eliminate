import os

import numpy as np
from pandas.tests.io.excel.test_xlsxwriter import xlsxwriter
from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
from pymoo.model.problem import Problem
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.visualization.scatter import Scatter
from optimization_utils import *

from run_exp_test import run_inference_for_optimization

#Nesma
#In this file optimization is done with three objectives to minimize WER and energy and maximize speedup
#Also WER<26 is used as a constraint to gurantee that the test wer is not > 27% and memory size is a constraint to guarantee compute-bound computation
# Each solution has 8 chromosome. They represent the precisions of the weight and activation of 8 layers
# The value of each chromosome is between 1 and 4 that are the encoding of the values 2-bit-integer, 4-bit-integer, 8-bit-integer, 16-bit-fixed-point
#The initial generation has 40 population, and each generation generates 10 new
# running for 40 generations would evaluate 430 solutions
# the size of the search space is 4096 solutions
# I change the value of the last chromosome before evaluation as it gives very bad solution that make the decoding very slow
#The output solutions is plotted in an png file and saved in an excel sheet file
#The output solutions are re-evaluated using the test set to know the testing WER that is also saved in the excel sheet.
#To compute the test WER. the inference has to run first with the optimization set to compute statistics required during running the testing inference.
#Speedup objective is inverted as the default in the library is to minimize objectives
#Speedup is computed by multiplying the number of MAC operations in each layer wit the speedup gained by the used precision over 16-bit on the same architecture
#Energy is computed by multiplying the number of bits loaded from SRAM and the number of computations for each precision by the energy cost of memory loading and the energy cost of each precision operation in pico joule respectively.
#TODO: change out_folder if required
out_folder='exp/TIMIT_SRU_fbank_dnn_6'
kaldi_path= os.environ['KALDI_ROOT']

class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=8, n_obj=3, n_constr=2, xl=2, xu=4, type_var=np.int,elementwise_evaluation=True)
        self.count=0
    def _evaluate(self, x  , out, *args, **kwargs):
        for i in range(8):
            x[i]=encode(x[i])
        xx=[x[0],x[1],x[2],x[3],x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[4],x[5],x[6],x[7],x[7]]
        wer, s = run_inference_for_optimization('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_SRU_fbank_dnn_6', xx,
                                                delta=[0, 0, 0, 0, 0, 0, 0, 0], n_layers=4, test=False, opt_index=4)
        mem_size =compute_mem(xx)


        sheet_log.write("B" + str(self.count + 2), str(wer))
      #  if wer<24:
      #      wer1 = run_inference_for_optimization(kaldi_path,out_folder,x[0],x[1],x[2],x[3],x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[4],x[5],x[6],x[7],x[7],False,1)
      #      wer2 =run_inference_for_optimization(kaldi_path,out_folder,x[0],x[1],x[2],x[3],x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[4],x[5],x[6],x[7],x[7],False,2)
      #      wer3=run_inference_for_optimization(kaldi_path,out_folder,x[0],x[1],x[2],x[3],x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[4],x[5],x[6],x[7],x[7],False,3)
      #      wer=max(wer1,wer2,wer3,wer)

        sheet_log.write("A" + str(self.count + 2), ','.join(str(z) for z in x))

        sheet_log.write("F" + str(self.count + 2), str(wer))
        f1 = wer
        f2=-(75900*get_speedup_silago(x[0])+ 844800*get_speedup_silago(x[1])+
        844800 * get_speedup_silago( x[2])+844800*get_speedup_silago(x[3])+
        281600 * get_speedup_silago( x[4])+281600 * get_speedup_silago( x[5])+
        281600 * get_speedup_silago( x[6])+2094400 * get_speedup_silago( x[7]))/5637500

        f3=(mem_size*8*1024*1024*0.08 + (75900*get_MAC_energy(x[0])+ 844800*get_MAC_energy(x[1])+
        844800 * get_MAC_energy( x[2])+844800*get_MAC_energy(x[3])+
        281600 * get_MAC_energy( x[4])+281600 * get_MAC_energy( x[5])+
        281600 * get_MAC_energy( x[6])+2094400 * get_MAC_energy( x[7])))/1000000
        sheet_log.write("G" + str(self.count + 2), str(mem_size))
        sheet_log.write("H" + str(self.count + 2), str(f2))
        sheet_log.write("I" + str(self.count + 2), str(f3))
        self.count+=1
        print(self.count)

        g2=(wer-24)/24

        g1=(mem_size-6)/6

        out["F"] = np.column_stack([f1, f2,f3])
        out["G"]=np.column_stack([g1,g2])

excel_log = xlsxwriter.Workbook('log_silago_new.xlsx')

sheet_log=excel_log.add_worksheet()
sheet_log.write('A1', 'solution')
sheet_log.write('B1', 'wer')
sheet_log.write('C1', 'wer1')
sheet_log.write('D1', 'wer2')
sheet_log.write('E1', 'wer3')
sheet_log.write('F1', 'max')
sheet_log.write('G1', 'size')
sheet_log.write('H1', 'speedup')
sheet_log.write('I1', 'energy')

method = NSGA2(
    pop_size=40,
    n_offsprings=10,
    sampling=get_sampling("int_random"),
    crossover=get_crossover("int_sbx", prob=1.0, eta=3.0),
    mutation=get_mutation("int_pm", eta=3.0),
    eliminate_duplicates=True
)


problem = MyProblem()
#method = np.load("checkpoint_silago_new.npy", allow_pickle=True).flatten()[0]

res = minimize(problem,
               method,
               ("n_gen", 15),
               seed=1,
               copy_algorithm=False,
               verbose=True)

np.save("checkpoint_silago_new", method)

plot = Scatter()
plot.add(res.F, color="red")
plot.save("pareto-silago_new")
excel_log.close()
excel = xlsxwriter.Workbook('optimize_silago_new.xlsx')
stat = excel.add_worksheet()
excel_init(stat)
excel_init_silago(stat)

for i in range (len(res.F)):
        # TODO otherwise decoding is very slow
    print("Hih")
    for j in range(8):
        res.X[i][j]=encode(res.X[i][j])
    if res.X[i][7] == 2:
        res.X[i][7] = 4
    xx = [res.X[i][0], res.X[i][1], res.X[i][2], res.X[i][3], res.X[i][0], res.X[i][1], res.X[i][2], res.X[i][3], res.X[i][4], res.X[i][5], res.X[i][6], res.X[i][4], res.X[i][5], res.X[i][6], res.X[i][7], res.X[i][7]]
    run_inference_for_optimization(kaldi_path,out_folder,xx,delta=[0, 0, 0, 0, 0, 0, 0, 0], n_layers=4, test=False, opt_index=4)
    real_wer=run_inference_for_optimization(kaldi_path,out_folder,xx, delta=[0, 0, 0, 0, 0, 0, 0, 0], n_layers=4, test=True)

    print_excel_silago(stat,i,res.F, res.X,real_wer,compute_mem([res.X[i][0], res.X[i][1], res.X[i][2], res.X[i][3], res.X[i][0], res.X[i][1], res.X[i][2], res.X[i][3], res.X[i][4], res.X[i][5], res.X[i][6], res.X[i][4], res.X[i][5],
                                   res.X[i][6], res.X[i][7], res.X[i][7]]))

excel.close()


