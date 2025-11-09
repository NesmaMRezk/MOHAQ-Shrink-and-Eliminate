from exp_sru import compute_sparsity_sru, compute_mem, compute_mem_6
from run_exp_test import run_inference_for_optimization_LSTM, compute_sparsity_lstm, compute_sparsity_gru, \
    run_inference_for_optimization_liGRU, run_inference_for_optimization_GRU, compute_sparsity_ligru, compute_mem_ligru, \
    run_inference_for_optimization

delta_lstm_4=[
[0, 8,8,8, 0, 8,8,8, 8],
[0, 8,8,8, 0, 8,8,8, 8],
[0, 8,8,8, 0, 8,8,8, 1],
[0, 8,8,8, 0, 8,8,8, 8],
[0, 8,8,8, 0, 8,8,8, 8],
[0, 0, 0, 0, 1,1,1,1, 1],#15.2, 51
[0, 0, 0, 0, 1,1,1,1, 1],#15.2, 51
[0, 0, 0, 0, 1,1,1,1, 1],#15.2, 51
[0, 0, 0, 0, 1,1,1,1, 1],#15.2, 51
[0, 0, 0, 0, 1,1,1,1, 1],#15.2, 51
#[0, 0, 0, 0, 0, 1,1,1, 0],#15,45
#[0, 1,1,1, 0, 0, 0, 0, 0], #15.5,42
]
exp_lstm_new1=[

  #[16,16,16,16,  16,16,16,16, 16,16,16,16, 16,16,16,16,   16,16],
  [8,8,8,8, 8,8,8,8, 8,8,8,8,  8,8,8,8,  8,8],
  #[4,4,4,4, 4,4,4,4, 4,4,4,4,  4,4,4,4, 4,4],
  #[2,2,2,2, 2,2,2,2, 2,2,2,2,  2,2,2,2, 2,4],
# [16,2,2,2,   16,2,2,2,  16,16,16,16,    16,16,16,16, 16,16],#M1
# [16,2,2,2,   16,2,2,2,  16,16,16,16,    16,16,16,16, 4,4],#M2
# [16,2,2,2,   16,2,2,2,  16,4,4,4,   16,16,16,16,  16,16],#M3
# [16,2,2,2,   16,2,2,2,  16,16,16,16,   16,4,4,4,  16,16],#M4
# [16,2,2,2,   2,2,2,2,  16,16,16,16,    16,16,16,16, 2,16],#M5
#[2,2,2,2,   2,2,2,2,  16,16,16,16,    16,16,16,16, 2,16],#M6

 [8,2,2,2,   2,2,2,2,  8,8,8,8,    8,8,8,8, 8,8],#M1
 [8,2,2,2,   2,2,2,2,  8,8,8,8,    8,8,8,8, 4,4],#M2
 [8,2,2,2,   2,2,2,2,  8,8,8,8,    8,8,8,8, 2,8],#M3
 [2,2,2,2,   2,2,2,2,  8,8,8,8,    8,8,8,8, 2,8],#M4

 [8,4,4,4,   4,4,4,4,  8,4,4,4,    4,4,4,4, 4,4],#M7
 #[8,2,2,2,   4,4,4,4,  8,4,4,4,    4,4,4,4, 4,4],#M8
 #[8,4,4,4,    4,2,2,2,  8,4,4,4,    4,4,4,4, 4,4],#M9
 #[4,2,2,2,    4,4,4,4,  8,4,4,4,    4,4,4,4, 4,4],#M10
 #[8,4,4,4,    2,2,2,2,  8,4,4,4,    4,4,4,4, 4,4],#M11
 #[8,4,4,4,    4,2,2,2,  8,4,4,4,    4,4,4,4, 2,4],#M12
 [8,2,2,2,    2,2,2,2,  8,4,4,4,    4,4,4,4, 4,4],#M13
 [8,2,2,2,    2,2,2,2,  8,4,4,4,    4,4,4,4, 2,4],#M14
 [4,2,2,2,    4,2,2,2,  4,4,4,4,    4,4,4,4, 2,4],#M15
[4,4,4,4,   4,4,4,4,  4,4,4,4,    4,4,4,4, 4,4],#M7
]

delta_lstm_8=[
[0, 0,0,0, 0,0,0,0, 0],
[0, 8,8,8,0,0,0,0, 0],
[0, 0,0,0, 0, 8,8,8, 0],
[0, 0,0,0, 0,0,0,0, 8],
[0, 8,8,8,0,0,0,0, 8],
[0, 0,0,0, 0, 8,8,8, 8],
[0, 8,8,8, 0, 8,8,8, 0],
[0, 8,8,8, 0, 8,8,8, 8],
[0, 0,0,0, 0,16,16,16,0],
[0, 16,16,16, 0, 0,0,0, 0],
[0, 0,0,0, 0,0,0,0, 16],
[0, 16,16,16, 0,0,0,0, 16],
[0, 0,0,0, 0, 16,16,16, 16],
[0, 16,16,16, 0, 16,16,16, 0],
[0, 16,16,16, 0, 16,16,16, 16],
[0, 32,32,32, 0,0,0,0, 0],
[0, 0,0,0, 0, 32,32,32, 0],
[0, 0,0,0, 0,0,0,0, 32],
[0, 32,32,32, 0,0,0,0, 32],
[0, 0,0,0, 0, 32,32,32, 32],
[0, 32,32,32, 0, 32,32,32, 0],
[0, 32,32,32, 0, 32,32,32, 32],

]
delta_gru_4=[
[0, 1,1,1,1, 0, 1,1,1,1, 1],
[0, 16,16,16,16, 0, 16,16,16,16, 16],
[0, 1,1,1,1, 0, 1,1,1,1, 1],
[0, 1,1,1,1, 0, 0,0,0,0, 0],
[0, 0,0,0,0, 0, 0,0,0,0, 1],
[0, 0,0,0,0, 0, 1,1,1,1, 1],
[0, 1,1,1,1, 0, 0,0,0,0, 1],
[0, 1,1,1,1, 0, 1,1,1,1, 0],
[0, 1,1,1,1, 0, 1,1,1,1, 1],
[0, 0,0,0,0, 0, 2,2,2,2, 0],
[0, 2,2,2,2, 0, 0,0,0,0, 0],
[0, 0,0,0,0, 0, 0,0,0,0, 2],
[0, 0,0,0,0, 0, 2,2,2,2, 2],
[0, 2,2,2,2, 0, 0,0,0,0, 2],
[0, 2,2,2,2, 0, 2,2,2,2, 0],
[0, 2,2,2,2, 0, 2,2,2,2, 2],
#[0, 0,0,0,0, 0, 1,1,1,1, 0],
#[0, 1,1,1,1, 0, 0,0,0,0, 0],
#[0, 1,1,1,1, 0, 1,1,1,1, 0],
#[0, 2,2,2,2, 0, 0,0,0,0, 0],
#[0, 0,0,0,0, 0, 2,2,2,2, 0],
#[0, 2,2,2,2, 0, 1,1,1,1, 0],
#[0, 1,1,1,1, 0, 2,2,2,2, 0],
#[0, 2,2,2,2, 0, 2,2,2,2, 0]
]

delta_gru_8=[
[0, 0,0,0,0, 0,0,0,0,0, 0],
[0, 8,8,8,8,0,0,0,0,0, 0],
[0, 0,0,0,0, 0, 8,8,8,8, 0],
[0, 0,0,0,0, 0,0,0,0,0, 8],
[0, 8,8,8,8,0,0,0,0,0, 8],
[0, 0,0,0, 0,0, 8,8,8,8, 8],
[0, 8,8,8,8, 0, 8,8,8,8, 0],
[0, 8,8,8,8, 0, 8,8,8,8, 8],
[0, 0,0,0,0, 0,16,16,16,16,0],
[0, 16,16,16,16, 0,0, 0,0,0, 0],
[0, 0,0,0,0, 0,0,0,0,0, 16],
[0, 16,16,16,16, 0,0,0,0,0, 16],
[0, 0,0,0, 0,0, 16,16,16,16, 16],
[0, 16,16,16,16, 0, 16,16,16,16, 0],
[0, 16,16,16,16, 0, 16,16,16,16, 16],
[0, 32,32,32,32, 0,0,0,0,0, 0],
[0, 0,0,0, 0,0, 32,32,32,32, 0],
[0, 0,0,0, 0,0,0,0,0,0, 32],
[0, 32,32,32,32, 0,0,0,0, 32],
[0, 0,0,0, 0,0, 32,32,32,32, 32],
[0, 32,32,32,32, 0, 32,32,32,32, 0],
[0, 32,32,32,32, 0, 32,32,32,32, 32],

[0, 48,48,48,48, 0,0,0,0,0, 0],
[0, 0,0,0, 0,0, 48,48,48,48, 0],
[0, 0,0,0, 0,0,0,0,0,0, 48],
[0, 48,48,48,48, 0,0,0,0,0, 48],
[0, 0,0,0, 0,0, 48,48,48,48, 48],
[0, 48,48,48,48, 0, 48,48,48,48, 0],
[0, 48,48,48,48, 0, 48,48,48,48, 48],
]
delta_sru=[
    [0,4,4,4,4,4,0,0,0,0,0,0],
 #   [0,0,0,0,0,0,0,0],

    [0,4,4,4,4,4,0,0,0,0,0,0],
    [0,4,4,4,4,4,0,0,0,0,0,0],
    [0,4,4,4,4,4,0,0,0,0,0,0],
    [0,4,4,4,4,4,0,0,0,0,0,0],

#    [0, 0, 0, 0, 0, 0, 0, 0],
#    [0, 0, 0, 0, 0, 0, 0, 0],
#    [0, 0, 0, 0, 0, 0, 0, 0],
#    [0, 0, 0, 0, 0, 0, 0, 0],

]
exp_sru=[
[8,8,8,8,8,8,  8,8,8,8,8,8,  8,8,8,8,8,  8,8,8,8,8, 8,8],
#[4,4,4,4,  4,4,4,4,  4,4,4,  4,4,4, 4,4],

[8,2,2,2,2,2,  8,8,8,8,8,8,  2,2,2,2,2,  8,8,8,8,8, 8,8],
[8,2,2,2,2,2,  8,8,8,8,8,8,  2,2,2,2,2,  8,8,8,8,8, 4,4],
[8,2,2,2,2,2,  8,8,8,8,8,8,  2,2,2,2,2,  8,8,8,8,8, 2,8],
[2,2,2,2,2,2,  8,8,8,8,8,8,  2,2,2,2,2,  8,8,8,8,8, 2,8],

[8,4,4,4,  8,4,4,4,  4,4,4,  4,4,4, 4,4],
[8,2,2,2,  8,4,4,4,  2,2,2,  4,4,4, 4,4],
[8,2,2,2,  8,4,4,4,  2,2,2,  4,4,4, 2,4],
[4,2,2,2,  4,4,4,4,  2,2,2,  4,4,4, 2,4]

]
exp=2
if exp==1:
      for i in range(9,len(delta_lstm_4)):
          wer,s=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_LSTM_fbank_c4',exp_lstm_new1[i],delta=delta_lstm_4[i], n_layers=4, test=False,opt_index=4,skip=1)
          wer, s = run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_LSTM_fbank_c4',
                                                       exp_lstm_new1[i], delta=delta_lstm_4[i], n_layers=4, test=True,
                                                        skip=1)
          print(compute_sparsity_lstm(list(map(float, s.split(","))),1))

          wer,s=run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_LSTM_fbank_c4',exp_lstm_new1[i],delta=delta_lstm_4[i], n_layers=4, test=False,opt_index=4,skip=2)
          wer, s = run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_LSTM_fbank_c4',
                                                       exp_lstm_new1[i], delta=delta_lstm_4[i], n_layers=4, test=True,
                                                        skip=2)
          print(compute_sparsity_lstm(list(map(float, s.split(","))),2))

elif exp==2:
     # for i in delta_lstm_8:
     #       print(i)
     # wer, s = run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_LSTM_fbank_c4',
     #                                                    [16,2,2,2, 16,2,2,2, 16, 8,8,8, 16, 8,8,8, 4,8],
     #                                                    delta=i, n_layers=4, test=False,skip=1)
     #       wer, s = run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_LSTM_fbank_c4',
     #                                                    [16,2,2,2, 16,2,2,2, 16, 8,8,8, 16, 8,8,8, 4,8],
     #                                                    delta=i, n_layers=4, test=True,skip=1)
     #       print(compute_sparsity_lstm(list(map(float, s.split(",")))))

       #     wer, s = run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_LSTM_fbank_c4',
       #                                                  [16, 2, 2, 2, 16, 2, 2, 2, 16, 8,8,8, 16, 8,8,8, 16, 16],
       #                                                  delta=i, n_layers=4, test=False,skip=1)
       #     wer, s = run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_LSTM_fbank_c4',
       #                                                  [16, 2, 2, 2, 16, 2, 2, 2, 16, 8,8,8, 16, 8,8,8, 16, 16],
       #                                                  delta=i, n_layers=4, test=True,skip=1)
       #     print(compute_sparsity_lstm(list(map(float, s.split(","))), 1))

       #     wer, s = run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_LSTM_fbank_c4',
       #                                                  [16, 2, 2, 2, 16, 2, 2, 2, 16, 8,8,8, 16, 8,8,8, 16, 16],
       #                                                  delta=i, n_layers=4, test=False,skip=2)
       #     wer, s = run_inference_for_optimization_LSTM('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_LSTM_fbank_c4',
       #                                                  [16, 2, 2, 2, 16, 2, 2, 2, 16, 8,8,8, 16, 8,8,8, 16, 16],
       #                                                  delta=i, n_layers=4, test=True,skip=2)
       #     print(compute_sparsity_lstm(list(map(float, s.split(","))), 2))

      #for i in delta_gru_8:
      #    print(i)
      #     wer, s = run_inference_for_optimization_liGRU('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_liGRU_fbank',
      #                                                 [16, 2, 2, 2,2, 16, 2,2, 2, 2, 16, 8,8, 8, 8, 16, 8,8, 8, 8, 16,16],
      #                                                 delta=i, n_layers=5, test=False)
      #     wer, s = run_inference_for_optimization_liGRU('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_liGRU_fbank',
      #                                                 [16, 2, 2, 2,2, 16, 2,2, 2, 2, 16, 8,8, 8, 8, 16, 8,8, 8, 8, 8,8],
      #                                                 delta=i, n_layers=5, test=True)
       #   print(compute_sparsity_ligru(list(map(float, s.split(",")))))

#          wer, s = run_inference_for_optimization_liGRU('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_liGRU_fbank',
#                                                       [16, 2, 2, 2,2, 16, 2,2, 2, 2, 16, 8,8, 8, 8, 16, 8,8, 8, 8, 16, 16],
#                                                       delta=i, n_layers=5, test=False, skip=1)
#          wer, s = run_inference_for_optimization_liGRU('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_liGRU_fbank',
#                                                       [16, 2, 2, 2,2, 16, 2,2, 2, 2, 16, 8,8, 8, 8, 16, 8,8, 8, 8, 16, 16],
#                                                       delta=i, n_layers=5, test=True, skip=1)
#          print(compute_sparsity_ligru(list(map(float, s.split(","))), 1))

#          wer, s = run_inference_for_optimization_liGRU('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_liGRU_fbank',
#                                                       [16, 2, 2, 2,2, 16, 2,2, 2, 2, 16, 8,8, 8, 8, 16, 8,8, 8, 8, 16, 16],
#                                                       delta=i, n_layers=5, test=False, skip=2)
#          wer, s = run_inference_for_optimization_liGRU('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_liGRU_fbank',
#                                                       [16, 2, 2, 2,2, 16, 2,2, 2, 2, 16, 8,8, 8, 8, 16, 8,8, 8, 8, 16, 16],
#                                                       delta=i, n_layers=5, test=True, skip=2)
#          print(compute_sparsity_ligru(list(map(float, s.split(","))), 2))
     exp_gru_4= [
          #[8,8,8,8,8, 8,8,8,8,8, 8,8,8,8,8,  8,8,8,8,8,  8,8],   #14.8,15
          [4,4,4,4,4,   4,4,4,4,4,   4,4,4,4,4,    4,4,4,4,4,  4,4],
          #[8,2,2,2,2,   2,2,2,2,2,  8,8,8,8, 8,    8,8,8,8,8,  8,8],#M1
          #[8,2,2,2,2,   2,2,2,2,2,  8,8,8,8,8,     8,8,8,8,8,  4,4],#M2
          #[8,2,2,2,2,   2,2,2,2,2,  8,8,8,8,8,     8,8,8,8,8,  2,8],#M3
          [2,2,2,2,2,   2,2,2,2,2,  8,8,8,8,8,     8,8,8,8,8,  2,8],#M4
          #[2,2,2,2,2,   2,2,2,2,2,  2,2,2,2,2,     2,2,2,2,2,  2,4],  # M5
          #[8,2,2,2,2,   2,2,2,2,2,  8,4,4,4,4,     4,4,4,4,4,  4,4],  # M6
          #[8,2,2,2,2,   2,2,2,2,2,  8,4,4,4,4,     4,4,4,4,4,  2,4],  # M7
          [4,2,2,2,2,   4,2,2,2,2,  4,4,4,4,4,     4,4,4,4,4,  2,4],  # M8
     ]
     delta_lstm_4 = [
     #    [0, 16, 16, 16,16, 16, 16, 16, 16, 16, 16, 16],
     #    [0, 16, 16, 16, 16, 16,16, 16, 16, 16, 16, 16],
      #   [0, 16, 16, 16, 16, 16, 16, 16,16, 16,  1],
     #    [0, 16, 16, 16, 16, 16, 16, 16, 16,16, 16, 16],
       #  [0, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,16],
     #    [0, 1,1,1,1,  1,1,1,1,1,1,  1],
         [0, 0,0,0,0, 0, 0,0,0,0, 0],
         [0, 0, 0, 0, 0, 0, 0,0,0,0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

         # [0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 0],
        # [0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4],
        # [0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4],

        # [0, 1,1,1,1, 0,0,0,0,0,0],
        # [0, 1,1,1,1, 0, 0, 0, 0, 0, 1],
        # [0, 0, 0, 0, 0, 0, 1,1,1,1,1],
        # [0, 1,1,1,1, 0, 1,1,1,1, 0],
        # [0, 0, 0, 0, 0, 0, 1,1,1,1, 0],
        # [0, 0, 0, 0, 0, 0, 0,0,0,0, 1],
        # [0, 1,1,1,1, 0, 1,1,1,1, 1],
        # [0, 2,2,2,2, 0, 0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0, 0, 2,2,2,2, 0],
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
        # [0, 2,2,2,2, 0, 2,2,2,2, 0],
         #[0, 16,16,16,16, 0, 0, 0, 0, 0, 0],
         #[0, 0, 0, 0, 0, 0, 16,16,16,16, 0],
       #  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16],
       #  [0, 16,16,16,16, 0, 16,16,16,16,16],
        # [0, 1, 1, 1, 1, 1, 1, 1, 1,1,1, 1],
         # [0, 0, 0, 0, 0, 1,1,1, 0],#15,45
         # [0, 1,1,1, 0, 0, 0, 0, 0], #15.5,42
         # [0, 1,1,1, 0, 1,1,1, 0], #15.9, 36
         # [0, 2,2,2, 0, 0, 0, 0, 0],# 19.5,38
         # [0, 0, 0, 0, 0, 2,2,2, 0],# 15.5, 43
         # [0, 2,2,2, 0, 1,1,1, 0], #20.3,33
         # [0, 1,1,1, 0, 2,2,2, 0],#17.3,35
         # [0, 2,2,2, 0,2,2,2, 0] #22.2, 32
     ]


     for i in range(len(delta_gru_4)):
     #   print(delta_gru_4[i])
         wer, s = run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_GRU_fbank',
                                                          exp_gru_4[i],
                                                       delta=delta_gru_4[i], n_layers=5, test=False,opt_index=4,skip=1)
         wer, s = run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_GRU_fbank',
                                                     exp_gru_4[i],
                                                     delta=delta_gru_4[i], n_layers=5, test=True, skip=1)

     #   wer, s = run_inference_for_optimization_liGRU('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_liGRU_fbank',
     #                                                 exp_gru_4[i],
     #                                              delta=delta_lstm_4[i], n_layers=5, test=True,skip=2)

     print(compute_sparsity_gru(list(map(float, s.split(","))), skip=2))
     #   print(compute_mem_ligru(exp_gru_4[i], 0))
     #   print(compute_mem_ligru(exp_gru_4[i], 1))
     #   print(compute_mem_ligru(exp_gru_4[i], 2))

     # wer, s = run_inference_for_optimization_liGRU('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_liGRU_fbank',
       #                                               exp_gru_4[i],
       #                                               delta=delta_lstm_4[i], n_layers=5, test=False, opt_index=4,
       #                                               skip=1)
       # wer, s = run_inference_for_optimization_liGRU('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_liGRU_fbank',
       #                                               exp_gru_4[i],
       #                                               delta=delta_lstm_4[i], n_layers=5, test=True, skip=1)

    #    print(compute_sparsity_ligru(list(map(float, "19.892329585552215,10.31326145529747,9.98674417734146,10.295326387882232,8.370417392253875,8.138636481761932,6.079545512795448,6.130681830644607,4.843181818723679,2.782954567670822,11.60346026479462".split(","))), skip=2))

        #print(compute_mem_ligru(i, 2))

        #wer, s = run_inference_for_optimization_liGRU('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_liGRU_fbank',
        #                                     exp_gru_4[i],
        #                                     delta=[0,0,0,0,0,0,0,0,0,0,0], n_layers=5, test=False, skip=2)
        #wer, s = run_inference_for_optimization_liGRU('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_liGRU_fbank',
        #                                     exp_gru_4[i],
        #                                     delta=delta_lstm_4[i], n_layers=5, test=True, skip=2)

        #print(compute_sparsity_ligru(list(map(float, s.split(","))), skip=2))
        #print(compute_mem_ligru(i, 2))
 #     for i in delta_gru_8:
 #         print(i)
 #         wer, s = run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_GRU_fbank',
 #                                                      [16, 16,16,16,16, 16,16,16,16,16, 16,16,16,16,16, 16, 16,16,16,16, 2,8],
 #                                                      delta=i, n_layers=5, test=False,skip=2)
 #         wer, s = run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_GRU_fbank',
 #                                                      [16, 16,16,16,16, 16, 16,16,16,16, 16,16,16,16,16, 16, 16,16,16,16, 2,8],
 #                                                      delta=i, n_layers=5, test=True,skip=2)

 #         print(compute_sparsity_gru(list(map(float, s.split(","))),skip=2))

     #     wer, s = run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_GRU_fbank',
     #                                                  [16, 2, 2, 2,2, 16, 2,2, 2, 2, 16, 8,8, 8, 8, 16, 8,8, 8, 8, 16, 16],
     #                                                  delta=i, n_layers=5, test=False, skip=1)
     #     wer, s = run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_GRU_fbank',
     #                                                  [16, 2, 2, 2,2, 16, 2,2, 2, 2, 16, 8,8, 8, 8, 16, 8,8, 8, 8, 16, 16],
     #                                                  delta=i, n_layers=5, test=True, skip=1)
     #     print(compute_sparsity_gru(list(map(float, s.split(","))), 1))

     #     wer, s = run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_GRU_fbank',
     #                                                  [16, 2, 2, 2,2, 16, 2,2, 2, 2, 16, 8,8, 8, 8, 16, 8,8, 8, 8, 16, 16],
     #                                                  delta=i, n_layers=5, test=False, skip=2)
     #     wer, s = run_inference_for_optimization_GRU('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_GRU_fbank',
     #                                                  [16, 2, 2, 2,2, 16, 2,2, 2, 2, 16, 8,8, 8, 8, 16, 8,8, 8, 8, 16, 16],
     #                                                  delta=i, n_layers=5, test=True, skip=2)
     #     print(compute_sparsity_gru(list(map(float, s.split(","))), 2))

if exp==3:
    for i in range(len(delta_sru)):
        print(i)
        wer, s = run_inference_for_optimization('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_SRU_fbank_dnn_l6', exp_sru[i], n_layers=6, delta=[0,0,0,0,0,0,0,0,0,0,0,0],
                                            test=False, opt_index=4)
        wer, s = run_inference_for_optimization('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_SRU_fbank_dnn_l6', exp_sru[i], n_layers=6, delta=[0,0,0,0,0,0,0,0,0,0,0,0], test=True)

        print(compute_mem_6(exp_sru[i]))
        print(compute_sparsity_sru(list(map(float, s.split(","))),n_layers=6))

     #   wer, s = run_inference_for_optimization('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_SRU_fbank_dnn_l6', exp_sru[i],
     #                                           n_layers=6, delta=delta_sru[i],
     #                                           test=False, opt_index=4)
     #   wer, s = run_inference_for_optimization('/usr/local/home/nesrez/kaldi', 'exp/TIMIT_SRU_fbank_dnn_l6', exp_sru[i],
     #                                           n_layers=6, delta=delta_sru[i], test=True)

     #   print(compute_mem_6(exp_sru[i]))
     #   print(compute_sparsity_sru(list(map(float, s.split(","))),n_layers=6))