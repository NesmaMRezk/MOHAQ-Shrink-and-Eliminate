import numpy as np
from pandas.tests.io.excel.test_xlsxwriter import xlsxwriter
from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
from pymoo.model.problem import Problem
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.visualization.scatter import Scatter
from optimization_utils import *
from run_exp_test import run_inference_for_optimization

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
def print_excel(stat,i,F,X):
    print(X[i])
    stat.write('B'+str(i+2),str(X[i][4]))
    stat.write('A'+str(i+2), str(X[i][0]))
    stat.write('D'+str(i+2), str(X[i][11]))
    stat.write('C'+str(i+2), str(X[i][8]))
    stat.write('F'+str(i+2), str(X[i][5]))
    stat.write('E'+str(i+2), str(X[i][1]))
    stat.write('G'+str(i+2), str(X[i][9]))
    stat.write('H'+str(i+2), str(X[i][12]))
    stat.write('I'+str(i+2), str(X[i][2]))
    stat.write('J'+str(i+2), str(X[i][6]))
    stat.write('K'+str(i+2), str(X[i][10]))
    stat.write('L'+str(i+2), str(X[i][13]))
    stat.write('M'+str(i+2), str(X[i][3]))
    stat.write('N'+str(i+2), str(X[i][7]))
    stat.write('O'+str(i+2), str(X[i][14]))
    stat.write('P'+str(i+2), str(X[i][15]))
    stat.write('Q'+str(i+2), str(F[i][0]))
    stat.write('R'+str(i+2), str(F[i][1]))
out_folder='exp/TIMIT_SRU_fbank_dnn_oo'
kaldi_path= os.environ['KALDI_ROOT']
distance_threshold=5
sheet_log=None
class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=16, n_obj=2, n_constr=2, xl=1, xu=4, type_var=np.int,elementwise_evaluation=True)
        self.count = 0
    def _evaluate(self, x  , out, *args, **kwargs):

        for i in range(16):
            x[i] = encode(x[i])
            # TODO otherwise decoding is very slow
        if x[15] == 2:
            x[15] = 4

        wer = run_inference_for_optimization(kaldi_path,out_folder,x,delta=[0,0,0,0,0,0,0,0], False)

        f2=-(75900*get_speedup_bitfusion(x[0],x[4])+ 844800*get_speedup_bitfusion(x[1],x[5])+
        844800 * get_speedup_bitfusion( x[2],x[6])+844800*get_speedup_bitfusion(x[3],x[7])+
        281600 * get_speedup_bitfusion( x[8],x[11])+281600 * get_speedup_bitfusion( x[9],x[12])+
        281600 * get_speedup_bitfusion( x[10],x[13])+2094400 * get_speedup_bitfusion( x[14],x[15]))/5637500/2
        mem_size = compute_mem([x[0], x[1], x[2], x[3], x[8], x[9], x[10], x[14]])
        wer_q=0
        beacon_index=0
        min_distance=0
        if wer<30 and wer>19 and f2<-10 and mem_size<2:
            beacon_index,min_distance=select_beacon(x)
            if min_distance>distance_threshold:
                wer_t = run_inference_for_optimization(kaldi_path,out_folder, x[0],x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11],  x[12], x[13],x[14],x[15], False)
                convert_to_beacon(wer_t, x, self.count)
                beacon_index, min_distance = select_beacon(x)
            wer_q= run_beacon(x,beacon_index)

            wer=min(wer,wer_q)
            print("min distance  "+str(min_distance))
        elif wer<24:
            wer1 = run_inference_for_optimization(kaldi_path, out_folder, x[0], x[1], x[2], x[3], x[4], x[5], x[6],
                                                  x[7], x[8], x[9], x[10], x[11],
                                                  x[12], x[13], x[14], x[15], False, 1)
            wer2 = run_inference_for_optimization(kaldi_path, out_folder, x[0], x[1], x[2], x[3], x[4], x[5], x[6],
                                                  x[7], x[8], x[9], x[10], x[11],
                                                  x[12], x[13], x[14], x[15], False, 2)
            wer3 = run_inference_for_optimization(kaldi_path, out_folder, x[0], x[1], x[2], x[3], x[4], x[5], x[6],
                                                  x[7], x[8], x[9], x[10], x[11],
                                                  x[12], x[13], x[14], x[15], False, 3)
            wer = max(wer1, wer2, wer3, wer)

        sheet_log.write("A" + str(self.count + 2), ','.join(str(z) for z in x))
        sheet_log.write("B" + str(self.count + 2), str(wer))
        sheet_log.write("C" + str(self.count + 2), str(wer_q))
        sheet_log.write("D" + str(self.count + 2), str(beacon_index))
        sheet_log.write("E" + str(self.count + 2), str(min_distance))
        sheet_log.write("G" + str(self.count + 2), str(mem_size))
        sheet_log.write("H" + str(self.count + 2), str(f2*-2))

        f1 = wer
        self.count += 1
        print(self.count)

        g1 = (wer - 24)/24
        g2= (mem_size - 2)/2


        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1,g2])


method = NSGA2(
    pop_size=40,
    n_offsprings=10,
    sampling=get_sampling("int_random"),
    crossover=get_crossover("int_sbx", prob=1.0, eta=3.0),
    mutation=get_mutation("int_pm", eta=3.0),
    eliminate_duplicates=True
)

problem = MyProblem()
# method = np.load("checkpoint.npy", allow_pickle=True).flatten()[0]
excel_log = xlsxwriter.Workbook('log_bitfusion_33.xlsx')

sheet_log=excel_log.add_worksheet()
sheet_log.write('A1', 'solution')
sheet_log.write('B1', 'wer')
sheet_log.write('C1', 'wer_retrained')
sheet_log.write('D1', 'index')
sheet_log.write('E1', 'distance')


# for i in range(6):
#   if i>-1:
#method = np.load("checkpoint_mem_bitfusion_beacons_2.npy", allow_pickle=True).flatten()[0]
res = minimize(problem,
               method,
               ('n_gen', 60),
               seed=1,
               copy_algorithm=False,
               verbose=True)
excel_log.close()
np.save("checkpoint_mem_bitfusion_beacons_33", method)

plot = Scatter()
plot.add(res.F, color="red")
plot.save("pareto-min_acc-min_mem_bitfusion_beacons_60_33")
excel = xlsxwriter.Workbook('optimize_acc_bitfusion_beacons-60-try_33.xlsx')
stat = excel.add_worksheet()
excel_init(stat)
excel_init_bitfusion(stat)
out_folder='exp/TIMIT_SRU_fbank_dnn_oo'
for i in range(len(res.F)):
    # TODO otherwise decoding is very slow
    for j in range(16):
        res.X[i][j] = encode(res.X[i][j])

    if res.X[i][15] == 2:
        res.X[i][15] = 4


    wer=run_inference_for_optimization(kaldi_path,out_folder,res.X[i][0], res.X[i][1], res.X[i][2], res.X[i][3], res.X[i][4], res.X[i][5],
                                   res.X[i][6], res.X[i][7], res.X[i][8], res.X[i][9], res.X[i][10], res.X[i][11],
                                   res.X[i][12],
                                   res.X[i][13], res.X[i][14], res.X[i][15], False)
    real_wer = run_inference_for_optimization(kaldi_path,out_folder,res.X[i][0], res.X[i][1], res.X[i][2], res.X[i][3], res.X[i][4],
                                              res.X[i][5], res.X[i][6], res.X[i][7], res.X[i][8], res.X[i][9],
                                              res.X[i][10], res.X[i][11], res.X[i][12],
                                              res.X[i][13], res.X[i][14], res.X[i][15], True)
    mem_size = compute_mem([res.X[i][0], res.X[i][1], res.X[i][2], res.X[i][3], res.X[i][8], res.X[i][9], res.X[i][10], res.X[i][14]])
    if wer < 30:
        beacon_index, b = select_beacon(res.X[i])
        print(beacon_index)
        wer_q = run_beacon(res.X[i], beacon_index,mono=True)
        real_wer_q = run_beacon(res.X[i], beacon_index, test=True)
        distance = compute_distance(res.X[i], beacons[beacon_index])
        print_excel_bitfusion(stat, i, res.F, res.X, real_wer,mem_size, wer_q, real_wer_q, str(beacon_index),str(distance))
    else:
        real_wer_q = -1
        wer_q = -1
        distance = -1
        print_excel_bitfusion(stat, i, res.F, res.X, real_wer,mem_size,wer_q, real_wer_q, str(distance))

    #print_excel_bitfusion(stat, i, res.F, res.X, real_wer,mem_size)

excel.close()

