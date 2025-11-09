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
out_folder='exp/TIMIT_SRU_fbank_dnn_l6'
kaldi_path= os.environ['KALDI_ROOT']
distance_threshold=5
sheet_log=None
class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=24, n_obj=3, n_constr=1, xl=1, xu=4, type_var=np.int,elementwise_evaluation=True)
        self.count = 0
    def _evaluate(self, x  , out, *args, **kwargs):

        play=False
        if not play:

            for i in range(24):
                x[i] = encode(x[i])
                # TODO otherwise decoding is very slow
            if x[23] == 2:
                x[23] = 4
            print("count")
            print(self.count)
            wer,s = run_inference_for_optimization(kaldi_path,out_folder,x,n_layers=6, test=False)

            f2=-(75900*get_speedup_bitfusion(x[0],x[6])+ 844800*get_speedup_bitfusion(x[1],x[7])+
            844800 * get_speedup_bitfusion( x[2],x[8])+844800*get_speedup_bitfusion(x[3],x[9])+844800*get_speedup_bitfusion(x[4],x[10])+844800*get_speedup_bitfusion(x[5],x[11])+
            281600 * get_speedup_bitfusion( x[12],x[17])+281600 * get_speedup_bitfusion( x[13],x[18])+
            281600 * get_speedup_bitfusion( x[14],x[19])+281600 * get_speedup_bitfusion( x[15],x[20])+281600 * get_speedup_bitfusion( x[16],x[21])+2094400 * get_speedup_bitfusion( x[22],x[23]))/7802300/2
            mem_size = compute_mem_6(x)
            f3=mem_size
            wer_q=0
            beacon_index=0
            min_distance=0
            wer_t = wer
            if self.count>40:
                wer_test, s = run_inference_for_optimization(kaldi_path, out_folder, x, n_layers=6, test=True)

                wer_1,wer_1_test=run_beacon_v2(x,1)
                print(self.count)
                wer_2,wer_2_test = run_beacon_v2(x, 2)
                if x[0]==2:
                    wer_3, wer_3_test = run_beacon_v2(x, 3)
                else:
                    wer_3 = 0
                    wer_3_test = 0
                # wer_4, wer_4_test = run_beacon_v2(x, 4)
                wer_5, wer_5_test = run_beacon_v2(x, 5)
               # wer_6, wer_6_test = run_beacon_v2(x, 6)
                if x[0]==16 and x[6]==16:
                    wer_9, wer_9_test = run_beacon_v2(x, 9)
                else:
                    wer_9 = 0
                    wer_9_test = 0

            else:
                wer_test=0
                wer_1=0
                wer_1_test=0
                wer_2=0
                wer_2_test=0
                wer_3=0
                wer_3_test=0
                wer_4 = 0
                wer_4_test = 0
                wer_5 = 0
                wer_5_test = 0
                wer_6 = 0
                wer_6_test = 0
                wer_9 = 0
                wer_9_test = 0
            distance_0=compute_distance_6(x,beacons_6[0])
            distance_1=compute_distance_6(x,beacons_6[1])
            distance_2 = compute_distance_6(x, beacons_6[2])
            distance_3 = compute_distance_6(x, beacons_6[3])
            distance_4 = compute_distance_6(x, beacons_6[4])
            distance_5 = compute_distance_6(x, beacons_6[5])
            distance_6 = compute_distance_6(x, beacons_6[6])
            distance_9 = compute_distance_6(x, beacons_6[9])
            distance_act_0=compute_distance_act_6(x,beacons_6[0])
            distance_act_1=compute_distance_act_6(x,beacons_6[1])
            distance_act_2 = compute_distance_act_6(x, beacons_6[2])
            distance_act_3 = compute_distance_act_6(x, beacons_6[3])
            distance_act_4 = compute_distance_act_6(x, beacons_6[4])
            distance_act_5 = compute_distance_act_6(x, beacons_6[5])
            distance_act_6 = compute_distance_act_6(x, beacons_6[6])
            distance_act_9 = compute_distance_act_6(x, beacons_6[9])

            sheet_log.write("A" + str(self.count + 2), ','.join(str(z) for z in x))
            sheet_log.write("B" + str(self.count + 2), str(wer_t))
            sheet_log.write("C" + str(self.count + 2), str(wer_test))
            sheet_log.write("E" + str(self.count + 2), str(distance_0))
            sheet_log.write("F" + str(self.count + 2), str(distance_act_0))
            sheet_log.write("G" + str(self.count + 2), str(wer_1))
            sheet_log.write("H" + str(self.count + 2), str(wer_1_test))
            sheet_log.write("I" + str(self.count + 2), str(distance_1))
            sheet_log.write("J" + str(self.count + 2), str(distance_act_1))
            sheet_log.write("K" + str(self.count + 2), str(wer_2))
            sheet_log.write("L" + str(self.count + 2), str(wer_2_test))
            sheet_log.write("M" + str(self.count + 2), str(distance_2))
            sheet_log.write("N" + str(self.count + 2), str(distance_act_2))
            sheet_log.write("O" + str(self.count + 2), str(wer_3))
            sheet_log.write("P" + str(self.count + 2), str(wer_3_test))
            sheet_log.write("Q" + str(self.count + 2), str(distance_3))
            sheet_log.write("R" + str(self.count + 2), str(distance_act_3))
            sheet_log.write("S" + str(self.count + 2), str(wer_5))
            sheet_log.write("T" + str(self.count + 2), str(wer_5_test))
            sheet_log.write("U" + str(self.count + 2), str(distance_5))
            sheet_log.write("V" + str(self.count + 2), str(distance_act_5))
            sheet_log.write("W" + str(self.count + 2), str(wer_9))
            sheet_log.write("X" + str(self.count + 2), str(wer_9_test))
            sheet_log.write("Y" + str(self.count + 2), str(distance_9))
            sheet_log.write("Z" + str(self.count + 2), str(distance_act_9))

            sheet_log.write("AA" + str(self.count + 2), str(mem_size))
            sheet_log.write("AB" + str(self.count + 2), str(f2*-2))

            f1 = wer
            self.count += 1
            print(self.count)

            g1 = (wer - 26)/26
            g2= (mem_size - 3)/3


            out["F"] = np.column_stack([f1, f2,f3])
            out["G"] = np.column_stack([g1])
        else:
            self.count += 1
            print(self.count)
            out["F"] = np.column_stack([20, -10])
            out["G"] = np.column_stack([20,3.5])

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

#sheet_log.write('A1', 'solution')
#sheet_log.write('B1', 'wer')
#sheet_log.write('C1', 'wer_retrained')
#sheet_log.write('D1', 'index')
#sheet_log.write('E1', 'distance')


# for i in range(6):
#   if i>-1:
n_gen=50
for i in range (48,n_gen-1,2):
    excel_log = xlsxwriter.Workbook('log_bitfusion_2n_'+str(i)+'.xlsx')
    sheet_log = excel_log.add_worksheet()


    if i>0:

        method, = np.load("checkpoint_mem_bitfusion_beacons_create_n2.npy", allow_pickle=True).flatten()
        method.has_terminated = False
    #excel_log = xlsxwriter.Workbook('log_acc__mem_lstm_5.xlsx')
    #sheet_log = excel_log.add_worksheet()

    res = minimize(problem,
                   method,

                   ('n_gen', i+2),
                   seed=1,
                   copy_algorithm=False,
                   verbose=True)

    np.save("checkpoint_mem_bitfusion_beacons_create_n2", method)
    excel_log.close()
    plot = Scatter()
    plot.add(res.F, color="red")

    #plot.save("pareto-min_acc-min_mem_bitfusion_beacons_"+str(i))


