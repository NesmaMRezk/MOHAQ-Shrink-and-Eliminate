import numpy as np
from pandas.tests.io.excel.test_xlsxwriter import xlsxwriter
from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
from pymoo.model.problem import Problem
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.visualization.scatter import Scatter


from optimization_utils import *
from run_exp_test import run_inference_for_optimization
retrain=False
def compute_mem_6(x):
    return (x[0] * 75900 + x[1] * 844800 + x[2] * 844800 + x[3] * 844800 +x[4] * 844800 + x[5] * 844800 + x[12] * 281600 + x[13] * 281600 + x[
        14] * 281600+ x[15] * 281600 + x[
        16] * 281600 + x[22] * 2094400 + 8800 * 16)/8/1024/1024

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
        super().__init__(n_var=24, n_obj=2, n_constr=2, xl=1, xu=4, type_var=np.int,elementwise_evaluation=True)
        self.count = 0
    def _evaluate(self, x  , out, *args, **kwargs):

        for i in range(24):
            x[i] = encode(x[i])
            # TODO otherwise decoding is very slow
        if x[23] == 2:
            x[23] = 4

        #wer,s = run_inference_for_optimization(kaldi_path,out_folder,x,n_layers=6,delta=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], test=False,opt_index=4)
        wer, s = run_inference_for_optimization(kaldi_path, out_folder, x, n_layers=6,
                                                delta=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], test=False, opt_index=4)
        f2=-(75900*get_speedup_bitfusion(x[0],x[6])+ 844800*get_speedup_bitfusion(x[1],x[7])+
        844800 * get_speedup_bitfusion( x[2],x[8])+844800*get_speedup_bitfusion(x[3],x[9])+844800*get_speedup_bitfusion(x[4],x[10])+844800*get_speedup_bitfusion(x[5],x[11])+
        281600 * get_speedup_bitfusion( x[12],x[17])+281600 * get_speedup_bitfusion( x[13],x[18])+
        281600 * get_speedup_bitfusion( x[14],x[19])+281600 * get_speedup_bitfusion( x[15],x[20])+281600 * get_speedup_bitfusion( x[16],x[21])+2094400 * get_speedup_bitfusion( x[22],x[23]))/7802300
        mem_size = compute_mem_6(x)
        wer_q=0
        beacon_index=0
        min_distance=0
      #  wer_t = wer
        index="base"
        if wer<30 and f2<-10 and mem_size<3 and retrain:
             #   convert_to_beacon_v2(wer_t, x, self.count)
             #   beacon_index, min_distance = select_beacon(x)
            if x[0]==2:
                wer_q,wer_tq = run_beacon_v2(x, 2,test=False)
                wer_q2, wer_tq2=0,0
                wer_q3, wer_tq3 = 0, 0
                index = "192"
            elif x[0]==16 and x[6]==16:
                wer_q,wer_tq = run_beacon_v2(x, 3,test=False)
                wer_q2, wer_tq2 = 0, 0
                wer_q3, wer_tq3 = 0, 0
                index = "156"
            else:
                wer_q,wer_tq= run_beacon_v2(x,1,test=False)
                index = "199"

            wer=min(wer,wer_q)
      #      print("min distance  "+str(min_distance))
      #  elif wer<24:
      #      wer1 = run_inference_for_optimization(kaldi_path, out_folder, x[0], x[1], x[2], x[3], x[4], x[5], x[6],
      #                                            x[7], x[8], x[9], x[10], x[11],
      #                                            x[12], x[13], x[14], x[15], False, 1)
      #      wer2 = run_inference_for_optimization(kaldi_path, out_folder, x[0], x[1], x[2], x[3], x[4], x[5], x[6],
      #                                            x[7], x[8], x[9], x[10], x[11],
      #                                            x[12], x[13], x[14], x[15], False, 2)
      #      wer3 = run_inference_for_optimization(kaldi_path, out_folder, x[0], x[1], x[2], x[3], x[4], x[5], x[6],
      #                                            x[7], x[8], x[9], x[10], x[11],
      #                                            x[12], x[13], x[14], x[15], False, 3)
      #      wer = max(wer1, wer2, wer3, wer)

        sheet_log.write("A" + str(self.count + 2), ','.join(str(z) for z in x))
        #sheet_log.write("B" + str(self.count + 2), str(wer))
        sheet_log.write("B" + str(self.count + 2), str(wer))

        sheet_log.write("F" + str(self.count + 2), index)
        sheet_log.write("G" + str(self.count + 2), str(min_distance))
        sheet_log.write("H" + str(self.count + 2), str(mem_size))
        sheet_log.write("I" + str(self.count + 2), str(f2))

        f1 = (wer-20)/20
        f2=(f2+40)/40
        self.count += 1
        print(self.count)

        g1 = (wer - 24)/24
        g2= (mem_size - 3)/3


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
#method = np.load("checkpoint.npy", allow_pickle=True).flatten()[0]

excel_log = xlsxwriter.Workbook('log_bitfusion_beacon_no_retrain6.xlsx')

sheet_log=excel_log.add_worksheet()
sheet_log.write('A1', 'solution')
sheet_log.write('B1', 'wer')
sheet_log.write('C1', 'wer_retrained')
sheet_log.write('D1', 'index')
sheet_log.write('E1', 'distance')


# for i in range(6):
#   if i>-1:
n_gen=90

for i in range (10000,10001):
    excel_log = xlsxwriter.Workbook('log_bitfusion_6p_enhanced6' + str(i) + '.xlsx')
    sheet_log = excel_log.add_worksheet()

    if i>0:

        method, = np.load("checkpoint_mem_bitfusion_beacons_enhanced6.npy", allow_pickle=True).flatten()
     #   method.has_terminated = False
    #excel_log = xlsxwriter.Workbook('log_acc__mem_lstm_5.xlsx')
    #sheet_log = excel_log.add_worksheet()

    res = minimize(problem,
                   method,
                   ('n_gen', i+2),
                   seed=1,
                   copy_algorithm=False,
                   verbose=True)
    np.save("checkpoint_mem_bitfusion_beacons_enhanced6", method)

    excel_log.close()


plot = Scatter()
plot.add(res.F, color="red")
plot.save("pareto-min_acc-min_mem_bitfusion_beacons_enhanced6")
excel = xlsxwriter.Workbook('optimize_acc_bitfusion_beacons_enhanced6.xlsx')
stat = excel.add_worksheet()
excel_init(stat)
#excel_init_bitfusion(stat)
out_folder='exp/TIMIT_SRU_fbank_dnn_l6'
print(len(res.F))
print(res.F)
for i in range(len(res.F)):
    # TODO otherwise decoding is very slow

    print(res.F[i][0]*20+20)
    print(res.F[i][1]*40-40)
for i in range(len(res.F)):
    # TODO otherwise decoding is very slow
    print(len(res.F))

    for j in range(24):
        res.X[i][j] = encode(res.X[i][j])

    if res.X[i][23] == 2:
        res.X[i][23] = 4

    print(res.X[i])
    mem_size = compute_mem_6(res.X[i])
    if retrain:# wer < 30 and res.F[i][1] < -10 and mem_size < 3:
      #  beacon_index, b = select_beacon_6(res.X[i])
      #  print(beacon_index)
      if res.X[i][0] == 2:
          wer_q,real_wer = run_beacon_v2(res.X[i], 2,test=True)
      elif res.X[i][0] == 16 and res.X[i][6] == 16:
          wer_q,real_wer = run_beacon_v2(res.X[i], 3,test=True)
      else:
          wer_q,real_wer = run_beacon_v2(res.X[i], 1,test=True)
    #  print_excel_bitfusion(stat, i, res.F, res.X, real_wer,mem_size, wer_q, real_wer_q, str(0),str(0))
    else:
        wer_q, real_wer = run_beacon_v2(res.X[i], 0, test=True)
        wer_q = -1
        distance = -1
     #   print_excel_bitfusion(stat, i, res.F, res.X, real_wer,mem_size,wer_q, real_wer_q, str(distance))
    stat.write("A" + str(i+1), ','.join(str(z) for z in res.X[i]))
    stat.write("B" + str(i+1), str(res.F[i][0]))
    stat.write("C" + str(i+1), str(wer_q))
    stat.write("D" + str(i+1), str(real_wer))

    stat.write("G" + str(i+1), str(mem_size))
    stat.write("H" + str(i+1), str(res.F[i][1]))

    #print_excel_bitfusion(stat, i, res.F, res.X, real_wer,mem_size)

excel.close()

