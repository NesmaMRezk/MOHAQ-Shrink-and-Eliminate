import os

import numpy as np
from pandas.tests.io.excel.test_xlsxwriter import xlsxwriter
from pymoo.algorithms.so_brkga import BRKGA
from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableCrossover, MixedVariableMutation
from pymoo.optimize import minimize
from pymoo.model.problem import Problem
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.visualization.scatter import Scatter

from run_exp_test import run_inference_for_optimization, run_inference_for_optimization_lstm_delta, \
    run_inference_for_optimization_LSTM
from optimization_utils import *
from optimization_utils import encode

#Nesma
n_layers= 4
n_precision= 4*n_layers + 2
n_delta= 2*n_layers + 1
model_per=[[],[0.01,0.35,0.17,0.17,0.3],[],[0.0036,0.17,0.17,0.17,0.08,0.08,0.08,0.08,0.15]]
Kaldi_path= os.environ['KALDI_ROOT']
first_gen=[17,35.5,40,16.6,18.3,35.8,17.2,17.2,19.7,17.5,14.7,16.8,19.4,16.6,15.5,15.6,18.1,35.8,36.3,14.9,16.1,17.3,17.7,37.9,15.7,34.5,15.8,16.3,38.2,17.1,16.3,16.1,16.1,36.2,38,19.2,17.8,14,18,32.2]
def encode_for_lstm(x,n=2):
    x1=[]
    x2=[]
    if n==2:
        margin=10
    elif n==4:
        margin=18
    if x[9]==0:
            x[9]=1
    for i in range(len(x)):
        if i<margin:
            x1.append(encode(x[i]+1))
        else:
            x2.append(x[i]/3)
#        elif i==10:
#            if x[i]==1:
#                x2.append(0)
#            elif x[i]==2:
#                x2.append(0.09)
#            elif x[i]==3:
#                x2.append(0.18)
#            else:
#                x2.append(0.36)
#        else:
#            if x[i]==1:
#                x2.append(0)
#            elif x[i]==2:
#                x2.append(0.03)
#            elif x[i]==3:
#                x2.append(0.06)
#            else:
#                x2.append(0.12)
    print(x1)
    print(x2)
    return x1,x2


class MyProblem(Problem):

    def __init__(self):

        super().__init__(n_var=n_precision, n_obj=2, n_constr=1, xl=0, xu=3, elementwise_evaluation=True)
        self.count=0
    def _evaluate(self, x  , out, *args, **kwargs):

        print(self.count)

        x1,x2=encode_for_lstm(x,n_layers)
        x2=[0,0,0,0,0,0,0,0,0]
        if self.count<40:
            wer=first_gen[self.count]
            sparsity_list=[100,100,100,100,100,100,100,100,100]
        else:
            wer,sparsity=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c4',x1,x2,n_layers,False)
            sparsity_list = list(map(float, sparsity.split(",")))

        f1 =wer


        f2=0
        excel_array=["B","D","F","H","J","L","N","P","R"]
        for i in range(n_precision//2):
            sparsity=100#sparsity_list[i]
            f2+=sparsity*model_per[n_layers-1][i]*x1[i]
            sheet_log.write(excel_array[i] + str(self.count + 2), str(sparsity))
        f2/=32
        #f2=(sparsity_list[0]*0.0025*x1[0]+sparsity_list[1]*0.48*x1[1]+sparsity_list[2]*0.23*x1[2]+sparsity_list[3]*0.23*x1[3]+sparsity_list[4]*0.04*x1[8])/32

        sheet_log.write("A" + str(self.count + 2), str(x1[0])+" * " +str(x1[8])+" * "+str(x2[0]))
        sheet_log.write("C" + str(self.count + 2), str(x1[1]) + " * " + str(x1[9]) + " * " + str(x2[1]))
        sheet_log.write("E" + str(self.count + 2), str(x1[2]) + " * " + str(x1[10]) + " * " + str(x2[2]))
        sheet_log.write("G" + str(self.count + 2), str(x1[3]) + " * " + str(x1[11]) + " * " + str(x2[3]))
        sheet_log.write("I" + str(self.count + 2), str(x1[4]) + " * " + str(x1[12]) + " * " + str(x2[4]))
        sheet_log.write("K" + str(self.count + 2), str(x1[5]) + " * " + str(x1[13]) + " * " + str(x2[5]))
        sheet_log.write("M" + str(self.count + 2), str(x1[6]) + " * " + str(x1[14]) + " * " + str(x2[6]))
        sheet_log.write("O" + str(self.count + 2), str(x1[7]) + " * " + str(x1[15]) + " * " + str(x2[7]))
        sheet_log.write("Q" + str(self.count + 2), str(x1[16]) + " * " + str(x1[17]) + " * " + str(x2[8]))



        sheet_log.write("S" + str(self.count + 2), str(wer))
        sheet_log.write("T" + str(self.count + 2), str(f2))
        print(wer)
        print(str(f2)+"%")
        self.count += 1


        g1=(wer-20)

        out["F"] = np.column_stack([f1,f2])
        out["G"]=np.column_stack([g1])

excel = xlsxwriter.Workbook('optimize_acc_mem_lstm_5.xlsx')
excel_log = xlsxwriter.Workbook('log_acc__mem_lstm_5.xlsx')

sheet_log=excel_log.add_worksheet()
stat = excel.add_worksheet()
mask=[]
for i in range(n_precision):
    mask.append("int")
#for i in range (n_delta):
#    mask.append("real")
sampling = MixedVariableSampling(mask, {
    "real": get_sampling("real_random"),
    "int": get_sampling("int_random")
})

crossover = MixedVariableCrossover(mask, {
    "real": get_crossover("real_sbx", prob=1.0, eta=3.0),
    "int": get_crossover("int_sbx", prob=1.0, eta=3.0)
})

mutation = MixedVariableMutation(mask, {
    "real": get_mutation("real_pm", eta=3.0),
    "int": get_mutation("int_pm", eta=3.0)
})
method = NSGA2(
    pop_size=40,
    n_offsprings=10,
    sampling=sampling,
    crossover=crossover,
    mutation=mutation,
    eliminate_duplicates=True
)


problem = MyProblem()
#method = np.load("checkpoint.npy", allow_pickle=True).flatten()[0]

n_gen=9
for i in range (n_gen-1):
    if i>0:
        method, = np.load("checkpoint_acc_mem_all_5.npy", allow_pickle=True).flatten()
        method.has_terminated = False
    #excel_log = xlsxwriter.Workbook('log_acc__mem_lstm_5.xlsx')
    #sheet_log = excel_log.add_worksheet()

    res = minimize(problem,
                   method,
                   ('n_gen', i+2),
                   seed=1,
                   copy_algorithm=False,
                   verbose=True)
    np.save("checkpoint_acc_mem_all_5", method)

    plot = Scatter()
    plot.add(res.F, color="red")
    plot.save("pareto_acc_mem_lstm_4")

excel_log.close()

plot = Scatter()
plot.add(res.F, color="red")
plot.save("pareto_acc_mem_lstm_4")
excel_init(stat)

for i in range (len(res.F)):
    print(res.F[i])
    print(res.X[i])
    x1, x2 = encode_for_lstm(res.X[i])
    print(x2)
    wer, s = run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_LSTM_fbank_c2', x1,x2, 2,False)
    sparsity_list = list(map(float, s.split(",")))
    # sparisty only on sru part
    f2 = (sparsity_list[0] * 0.0025 * x1[0] + sparsity_list[1] * 0.48 * x1[1] + sparsity_list[2] * 0.23 * x1[2] +
          sparsity_list[3] * 0.23 * x1[4] + sparsity_list[4] * 0.04 * x1[8]) / 32
    print(f2)
    x2 = [0, 0, 0, 0, 0]

    wer_no_delta, s= run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_LSTM_fbank_c2', x1, x2, 2,
                                               False)
 #   wer_no_delta_t, s= run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_LSTM_fbank_c2', x1, x2, 2,
 #                                              True)
    x1, x2 = encode_for_lstm(res.X[i])
    x1 = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16]
    wer_fixed,s = run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_LSTM_fbank_c2', x1, x2, 2,
                                               False)
    #wer_fixed_t,s = run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_LSTM_fbank_c2', x1, x2, 2,
    #                                           True)
    #real_wer,=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi','exp/TIMIT_LSTM_fbank_c2',x1,x2,2, True)
    stat.write('A' + str(i + 2),  ','.join(str(z) for z in x1))
    stat.write('B' + str(i + 2),  ','.join(str(z) for z in x2))
    stat.write('C' + str(i + 2), str(res.F[i][0]))
    stat.write('D' + str(i + 2), str(res.F[i][1]))
    stat.write('E' + str(i + 2), str(wer_no_delta))
    stat.write('F' + str(i + 2), str(wer_fixed))
    stat.write('G' + str(i + 2), str(wer_no_delta_t))
    stat.write('H' + str(i + 2), str(wer_fixed_t))

    print("done")
excel.close()


