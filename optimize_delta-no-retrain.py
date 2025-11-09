import os

import numpy as np
from pandas.tests.io.excel.test_xlsxwriter import xlsxwriter
from pymoo.algorithms.so_brkga import BRKGA
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

class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=8, n_obj=2, n_constr=1, xl=0, xu=8, type_var=np.int,elementwise_evaluation=True)
        self.count=0
    def _evaluate(self, x  , out, *args, **kwargs):

        print(self.count)


        wer,sparsity=run_inference_for_optimization('/usr/local/home/nesrez/kaldi','exp/TIMIT_SRU_fbank_dnn_oo',16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,x,False,1)
        f1 =wer
        sparsity_list=list(map(float, sparsity.split(",")))
        # sparisty only on sru part
        f2=sparsity_list[0]*0.014+sparsity_list[1]*0.051+sparsity_list[2]*0.15+sparsity_list[3]*0.051+sparsity_list[4]*0.15+sparsity_list[5]*0.051+sparsity_list[6]*0.15
        sheet_log.write("A" + str(self.count + 2), ','.join(str(z) for z in x))
        sheet_log.write("B" + str(self.count + 2), str(wer))
       # sheet_log.write("C" + str(self.count + 2), str(wer_test))
        sheet_log.write("D" + str(self.count + 2), str(f2))

        self.count += 1


        g1=(wer-24)

        out["F"] = np.column_stack([f1,f2])
        out["G"]=np.column_stack([g1])

excel = xlsxwriter.Workbook('optimize_acc_mem_delta_all.xlsx')
excel_log = xlsxwriter.Workbook('log_acc__mem_delta_all.xlsx')

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
#method = np.load("checkpoint_acc_mem_delta_all.npy", allow_pickle=True).flatten()[0]
res = minimize(problem,
               method,
               ('n_gen', 12),
               seed=1,
               copy_algorithm=False,
               verbose=True)
np.save("checkpoint_acc_mem_delta_all", method)
excel_log.close()
plot = Scatter()
plot.add(res.F, color="red")
plot.save("pareto_acc_mem_delta_all")
excel_init(stat)

for i in range (len(res.F)):

    wer = run_inference_for_optimization('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_SRU_fbank_dnn_oo',16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,res.X[i], False)
    real_wer=run_inference_for_optimization('/usr/local/home/nesrez/kaldi','exp/TIMIT_SRU_fbank_dnn_oo',16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,res.X[i], True)
    print_excel_delta(stat, i, res.F, res.X, real_wer)


    print("done")
excel.close()


