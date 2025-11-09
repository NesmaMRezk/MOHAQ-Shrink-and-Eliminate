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
out_folder='exp/TIMIT_SRU_fbank_dnn_6'
kaldi_path= os.environ['KALDI_ROOT']
distance_threshold=5
sheet_log=None
class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=16, n_obj=2, n_constr=2, xl=1, xu=4, type_var=np.int,elementwise_evaluation=True)
        self.count = 0
    def _evaluate(self, x  , out, *args, **kwargs):

        play=False
        if not play:

            for i in range(16):
                x[i] = encode(x[i])
                # TODO otherwise decoding is very slow
            if x[15] == 2:
                x[15] = 4

            wer,s = run_inference_for_optimization(kaldi_path,out_folder,x,delta=[0,0,0,0,0,0,0,0],n_layers=4, test=True)

            f2 = -(75900 * get_speedup_bitfusion(x[0], x[4]) + 844800 * get_speedup_bitfusion(x[1], x[5]) +
                   844800 * get_speedup_bitfusion(x[2], x[6]) + 844800 * get_speedup_bitfusion(x[3], x[7]) +
                   281600 * get_speedup_bitfusion(x[8], x[11]) + 281600 * get_speedup_bitfusion(x[9], x[12]) +
                   281600 * get_speedup_bitfusion(x[10], x[13]) + 2094400 * get_speedup_bitfusion(x[14],
                                                                                            x[15])) / 5637500 / 2
            mem_size = compute_mem(x)
            wer_q=0
            beacon_index=0
            min_distance=0
            wer_test = wer
            if self.count>100:
#                wer_test, s = run_inference_for_optimization(kaldi_path, out_folder, x, n_layers=4, test=True)
                #wer_1=run_beacon(x,1,test=True)
                if x[0]==2:

                    wer_1_test = run_beacon(x, 3, test=True)
                    wer_2, wer_2_test=0,0
                    wer_3, wer_3_test = 0, 0
                elif x[0]==16 and x[4]==16:

                    wer_1_test = run_beacon(x, 4, test=True)
                    wer_2, wer_2_test=0,0
                    wer_3, wer_3_test=0,0
                else:

                    wer_1_test = run_beacon(x, 1, test=True)

               # wer_2 = run_beacon(x, 2,test=False)
                   # wer_2_test = run_beacon(x, 2, test=True)

               # wer_3= run_beacon(x, 3,test=False)
                  #  wer_3_test=run_beacon(x, 3,test=True)

            else:

                wer_1=0
                wer_1_test=0
                wer_2=0
                wer_2_test=0
                wer_3=0
                wer_3_test=0
            distance_0=compute_distance(x,beacons[0])
            distance_1=compute_distance(x,beacons[1])
            distance_2 = compute_distance(x, beacons[2])
            distance_3 = compute_distance(x, beacons[3])

            sheet_log.write("A" + str(self.count + 2), ','.join(str(z) for z in x))
            sheet_log.write("B" + str(self.count + 2), str(wer_test))
            sheet_log.write("C" + str(self.count + 2), str(wer_test))
            sheet_log.write("D" + str(self.count + 2), str(distance_0))
#            sheet_log.write("E" + str(self.count + 2), str(wer_1))
            sheet_log.write("F" + str(self.count + 2), str(wer_1_test))
            sheet_log.write("G" + str(self.count + 2), str(distance_1))
 #           sheet_log.write("H" + str(self.count + 2), str(wer_2))
            #sheet_log.write("I" + str(self.count + 2), str(wer_2_test))
            #sheet_log.write("J" + str(self.count + 2), str(distance_2))
  #          sheet_log.write("K" + str(self.count + 2), str(wer_3))
            #sheet_log.write("L" + str(self.count + 2), str(wer_3_test))
            #sheet_log.write("M" + str(self.count + 2), str(distance_3))

            sheet_log.write("N" + str(self.count + 2), str(mem_size))
            sheet_log.write("O" + str(self.count + 2), str(f2*-2))

            f1 = wer
            self.count += 1
            print(self.count)

            g1 = (wer - 24)/24
            g2= (mem_size - 2)/2


            out["F"] = np.column_stack([f1, f2])
            out["G"] = np.column_stack([g1,g2])
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
for i in range (14,n_gen-1):
    excel_log = xlsxwriter.Workbook('log_bitfusion_4_dec_'+str(i)+'.xlsx')
    sheet_log = excel_log.add_worksheet()


    if i>0:

        method, = np.load("checkpoint_mem_bitfusion_beacons_create_4_dec.npy", allow_pickle=True).flatten()
        method.has_terminated = False
    #excel_log = xlsxwriter.Workbook('log_acc__mem_lstm_5.xlsx')
    #sheet_log = excel_log.add_worksheet()

    res = minimize(problem,
                   method,

                   ('n_gen', i+5),
                   seed=1,
                   copy_algorithm=False,
                   verbose=True)

    np.save("checkpoint_mem_bitfusion_beacons_create_4_dec", method)
    excel_log.close()
    plot = Scatter()
    plot.add(res.F, color="red")

    #plot.save("pareto-min_acc-min_mem_bitfusion_beacons_"+str(i))


