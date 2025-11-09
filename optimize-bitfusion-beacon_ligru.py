import numpy as np
from pandas.tests.io.excel.test_xlsxwriter import xlsxwriter
from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
from pymoo.model.problem import Problem
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.visualization.scatter import Scatter


from optimization_utils import *
from run_exp_test import run_inference_for_optimization
retrain=True
first_gen=[37.8,22.7,23.3,16.7,14.5,15.6,33.5,20.2,15.8,14,38.6,37.5,13.5,14.1,21.4,37.3,15.1,15.3,13.5,13.9,13.5,15.5,33.8,20.9,14,13.7,14.9,36.1,12.9,24.3,46.3,14.5,23.2,17.7,14.9,14.7,14.2,13.6,33.2,13.9]

def compute_mem_ligru(x,skip=0):
   if skip==0:
       return (x[0]*25300 + x[1]*1210000 + x[2]*1210000 +x[3]*1210000+x[4]*1210000+ x[5]*605000 +x[6]* 605000+x[7]*605000 +x[8]*605000+x[9]*605000 +x[20]*2129600 + 5500*16)/8/1024/1024
   elif skip==1:
       return (x[0]*25300 + x[1]*1210000 + x[2]*1210000 +x[3]*1210000 + x[4] * 1210000+ 2/3*(x[5]*605000 +x[6]*605000 +x[7]*605000 +x[8]*605000+x[9]*605000 )+16/3*(605000*5)+x[20]*2129600 +5500 *16)/8/1024/1024
   elif skip == 2:
       return (x[0] * 25300 + x[1] * 1210000 + x[2] * 1210000 + x[3] * 1210000+ x[4] * 1210000 + 2 / 3 * (
                   x[5] * 605000 + x[6] * 605000+ x[7] * 605000) + x[8] * 605000 + x[9] * 605000 + 16 / 3 * (605000 * 3) + x[
                   20] * 2129600 + 5500 * 16) / 8 / 1024 / 1024

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
out_folder='exp/TIMIT_liGRU_fbank'
kaldi_path= os.environ['KALDI_ROOT']
distance_threshold=5
sheet_log=None
class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=14, n_obj=2, n_constr=2, xl=1, xu=4, type_var=np.int,elementwise_evaluation=True)
        self.count = 0
    def _evaluate(self, x  , out, *args, **kwargs):
        print(self.count)

        for i in range(14):
            x[i] = encode(x[i])
            # TODO otherwise decoding is very slow
        if x[13] == 2:
            x[13] = 4

        y=[x[0],x[2],x[3],x[4],x[5], x[1],x[2],x[3],x[4],x[5], x[6],x[8],x[9],x[10],x[11], x[7],x[8],x[9],x[10],x[11], x[12],x[13] ]
        x=y
        #wer,s = run_inference_for_optimization(kaldi_path,out_folder,x,n_layers=6,delta=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], test=False,opt_index=4)
        if self.count<40:
            wer=first_gen[self.count]
        else:
            wer, s = run_inference_for_optimization_liGRU(kaldi_path, out_folder, x, n_layers=5,delta=[0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0], test=False, opt_index=4,skip=1)
        f2 = -(get_speedup_bitfusion(x[0],x[10]) * 0.0019 + get_speedup_bitfusion(x[1],x[11])* 0.0927 + get_speedup_bitfusion(x[2],x[12]) * 0.0927 \
             + get_speedup_bitfusion(x[3],x[13]) * 0.0927 + get_speedup_bitfusion(x[4],x[14])* 0.0927 + 1/2*(get_speedup_bitfusion(x[5],x[15]) * 0.0927 \
            + get_speedup_bitfusion(x[6],x[16]) * 0.0927 + get_speedup_bitfusion(x[7],x[17]) * 0.0927 + get_speedup_bitfusion(x[8],x[18]) * 0.0927+\
             get_speedup_bitfusion(x[9],x[19]) * 0.0927 ) + 1/2*0.0927*5+ get_speedup_bitfusion(x[20],x[21]) * 0.163 + 0.0002)
        mem_size = compute_mem_ligru(x,skip=1)
        wer_q=0
        beacon_index=0
        min_distance=0
      #  wer_t = wer
        index="base"
        if wer<30 and f2<-10 and mem_size<4.5 and retrain:
             #   convert_to_beacon_v2(wer_t, x, self.count)
             #   beacon_index, min_distance = select_beacon(x)
            if x[0]==2:
                wer_q,wer_tq = run_beaconG(x, 3,test=False)
                wer_q2, wer_tq2=0,0
                wer_q3, wer_tq3 = 0, 0
                index = "192"
            elif x[0]==16 and x[10]==16:
                wer_q,wer_tq = run_beaconG(x, 2,test=False)
                wer_q2, wer_tq2 = 0, 0
                wer_q3, wer_tq3 = 0, 0
                index = "156"
            else:
                wer_q,wer_tq= run_beaconG(x,1,test=False)
            #    wer_q2, wer_tq2 = run_beacon_v2(x, 4, test=True)
            #    wer_q3, wer_tq3 = run_beacon_v2(x, 5, test=True)
                index = "152"

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
        sheet_log.write("B" + str(self.count + 2), str(x[0]))
        sheet_log.write("C" + str(self.count + 2), str(x[10]))
        sheet_log.write("D" + str(self.count + 2), str(x[5]))
        sheet_log.write("E" + str(self.count + 2), str(x[15]))
        sheet_log.write("F" + str(self.count + 2), str(wer))
        sheet_log.write("G" + str(self.count + 2), str(wer_q))
        sheet_log.write("H" + str(self.count + 2), index)
        sheet_log.write("J" + str(self.count + 2), str(mem_size))
        sheet_log.write("K" + str(self.count + 2), str(f2))

        f1 = wer
        f2=(f2+40)/40
        self.count += 1

        g1 = (wer - 24)/24
        g2= (mem_size - 4.5)/4.5


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

excel_log = xlsxwriter.Workbook('log_bitfusion_beacon_ligru_beacon.xlsx')

sheet_log=excel_log.add_worksheet()
sheet_log.write('A1', 'solution')
sheet_log.write('B1', 'wer')
sheet_log.write('C1', 'wer_retrained')
sheet_log.write('D1', 'index')
sheet_log.write('E1', 'distance')


# for i in range(6):
#   if i>-1:
n_gen=50
for i in range (49
        ,n_gen):
  #  excel_log = xlsxwriter.Workbook('log_bitfusion_6p_' + str(i) + '.xlsx')
  #  sheet_log = excel_log.add_worksheet()

    if i>0:

        method, = np.load("checkpoint_mem_bitfusion_ligru_beacon.npy", allow_pickle=True).flatten()
        #method.has_terminated = False
    #excel_log = xlsxwriter.Workbook('log_acc__mem_lstm_5.xlsx')
    #sheet_log = excel_log.add_worksheet()

    res = minimize(problem,
                  method,
                   ('n_gen', i+2),
                   seed=1,
                   copy_algorithm=False,
                   verbose=True)
    np.save("checkpoint_mem_bitfusion_ligru_beacon", method)

    excel_log.close()

plot = Scatter()
plot.add(res.F, color="red")
plot.save("pareto-min_acc-checkpoint_mem_bitfusion_ligru_beacon")
excel = xlsxwriter.Workbook('optimize_checkpoint_mem_bitfusion_ligru_beacon.xlsx')
stat = excel.add_worksheet()
excel_init(stat)
#excel_init_bitfusion(stat)
out_folder='exp/TIMIT_liGRU_fbank'
for i in range(len(res.F)):
    print(res.F[i])
    print(res.X[i])
for i in range(len(res.F)):
    # TODO otherwise decoding is very slow
    for j in range(14):
        res.X[i][j] = encode(res.X[i][j])

    if res.X[i][13] == 2:
        res.X[i][13] = 4
    y=[res.X[i][0],res.X[i][2],res.X[i][3],res.X[i][4],res.X[i][5], res.X[i][1],res.X[i][2],res.X[i][3],res.X[i][4],res.X[i][5], res.X[i][6],res.X[i][8],res.X[i][9],res.X[i][10],res.X[i][11], res.X[i][7],res.X[i][8],res.X[i][9],res.X[i][10],res.X[i][11], res.X[i][12],res.X[i][13] ]
 #   res.X[i]=y
    mem_size = compute_mem_ligru(y,skip=1)
    if  retrain:# wer < 30 and res.F[i][1] < -10 and mem_size < 3:
      #  beacon_index, b = select_beacon_6(res.X[i])
      #  print(beacon_index)
        if y[0] == 2:
            wer_q,real_wer = run_beaconG(y, 3,test=True)
        elif y[0] == 16 and y[10] == 16:
            wer_q,real_wer = run_beaconG(y, 2,test=True)
        else:
            wer_q,real_wer = run_beaconG(y, 1,test=True)
    #  print_excel_bitfusion(stat, i, res.F, res.X, real_wer,mem_size, wer_q, real_wer_q, str(0),str(0))
    else:
        wer_q,real_wer = run_beaconG(y, 0,test=True)
        distance = -1
     #   print_excel_bitfusion(stat, i, res.F, res.X, real_wer,mem_size,wer_q, real_wer_q, str(distance))
    stat.write("A" + str(i+1), ','.join(str(z) for z in y))
    stat.write("B" + str(i+1), str(res.F[i][0]))
    stat.write("C" + str(i+1), str(wer_q))
    stat.write("D" + str(i+1), str(real_wer))

    stat.write("G" + str(i+1), str(mem_size))
    stat.write("H" + str(i+1), str(res.F[i][1]*40-40))

    #print_excel_bitfusion(stat, i, res.F, res.X, real_wer,mem_size)

excel.close()

